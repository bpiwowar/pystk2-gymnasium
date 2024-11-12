from functools import partial, partialmethod
import logging
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
import sys
from typing import List, Optional
import pystk2

logger = logging.getLogger("pystk2_gymnasium.mp")


class PySTKRemoteProcess:
    world: Optional[pystk2.WorldState] = None
    track: Optional[pystk2.Track] = None
    race: Optional[pystk2.Race] = None

    def __init__(self, with_graphics: bool):
        pystk2.init(
            pystk2.GraphicsConfig.hd()
            if with_graphics
            else pystk2.GraphicsConfig.none()
        )

    @staticmethod
    def run(with_graphics: bool, level, pipe: Connection):
        logging.basicConfig(level=level)
        stk = PySTKRemoteProcess(with_graphics)

        while True:
            command = pipe.recv()
            if command is None:
                # We stop if the command is None
                pipe.send(None)
                sys.exit()

            logging.debug(
                "Received command %s, args=%s, kwargs=%s",
                command.func,
                command.args,
                command.keywords,
            )
            assert isinstance(command, partialmethod)

            result = command.func(stk, *command.args, **command.keywords)
            logging.debug("Sending result %s", result)
            pipe.send(result)

    def list_tracks(self) -> List[str]:
        return pystk2.list_tracks(pystk2.RaceConfig.RaceMode.NORMAL_RACE)

    def close(self):
        super().close()
        if self.race is not None:
            self.race.stop()
            self.race = None

    def warmup_race(self, config) -> pystk2.Track:
        """Creates a new race and step until the first move"""
        if self.race is not None:
            self.race.stop()
            self.race = None

        self.race = pystk2.Race(config)

        # Start race
        self.race.start()
        self.world = pystk2.WorldState()
        track = pystk2.Track()

        while True:
            self.race.step()
            self.world.update()
            if self.world.phase == pystk2.WorldState.Phase.READY_PHASE:
                break

        track.update()
        return track

    def get_world(self):
        if self.world is None:
            return Exception("Cannot get world state since race has not been started")
        self.world.update()
        return self.world

    def race_step(self, *args):
        if self.race is None:
            return Exception("Cannot step since race has not been started")
        return self.race.step(*args)

    def get_kart_action(self, kart_ix):
        if self.race is None:
            return Exception("Cannot step since race has not been started")
        return self.race.get_kart_action(kart_ix)


class PySTKProcess:
    COUNT = 0

    def __init__(self, with_graphics: bool):
        self.pipe, remote_pipe = Pipe(True)
        PySTKProcess.COUNT += 1
        logging.info(f"Creating pystk-{PySTKProcess.COUNT}")
        self.process = Process(
            name=f"pystk-{PySTKProcess.COUNT}",
            target=PySTKRemoteProcess.run,
            args=[with_graphics, logging.getLogger().level, remote_pipe],
            daemon=True,
        )
        self.process.start()

    def _run(self, method, *args, **kwargs):
        if method:
            method = partialmethod(method, *args, **kwargs)
        else:
            assert len(args) == 0 and len(kwargs) == 0

        self.pipe.send(method)
        result = self.pipe.recv()
        logging.debug("Got %s", result)
        if isinstance(result, Exception):
            raise result
        return result

    def __del__(self):
        logging.debug("Stopping the process")
        try:
            self.close()
        except BrokenPipeError:
            # Ignores when the process was already stopped
            pass

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self._run(None)
                self.process.kill()
                self.process = None

    def __getattr__(self, name):
        return partial(self._run, getattr(PySTKRemoteProcess, name))
