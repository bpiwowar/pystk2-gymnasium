"""Shared protocol for race client/server communication over ZMQ."""

import pickle

PROTOCOL_VERSION = 1

# Message types: client → server
MSG_INIT = "init"
MSG_SPACES = "spaces"
MSG_STEP = "step"
MSG_CLOSE = "close"

# Message types: server → client
MSG_INIT_RESPONSE = "init_response"
MSG_SPACES_RESPONSE = "spaces_response"
MSG_STEP_RESPONSE = "step_response"
MSG_CLOSE_RESPONSE = "close_response"
MSG_ERROR = "error"


def send_msg(socket, msg: dict):
    """Serialize and send a message over a ZMQ socket."""
    socket.send(pickle.dumps(msg))


def recv_msg(socket) -> dict:
    """Receive and deserialize a message from a ZMQ socket."""
    return pickle.loads(socket.recv())
