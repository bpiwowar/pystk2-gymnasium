import gymnasium
import numpy as np
import numpy.testing
import pystk2
from pystk2_gymnasium.utils import Discretizer, rotate, rotate_batch
from pystk2_gymnasium.envs import STKRaceEnv


def test_rotation():
    env = None
    try:
        env = STKRaceEnv()
        env.initialize(False)
        env.config = pystk2.RaceConfig(num_kart=1, track="lighthouse")
        env.warmup_race()
        world = env.world_update(False)

        kart = world.karts[0]
        np.allclose(kart.velocity_lc, rotate(kart.velocity, kart.rotation))
    finally:
        if env is not None:
            env.close()


def test_rotate_batch_consistency():
    """Test that rotate_batch produces same results as multiple rotate calls."""
    np.random.seed(42)

    # Generate random quaternion (normalized)
    q = np.random.randn(4).astype(np.float32)
    q = q / np.linalg.norm(q)

    # Generate random vectors
    num_vectors = 100
    vectors = np.random.randn(num_vectors, 3).astype(np.float32)

    # Compute using batch function
    batch_result = rotate_batch(vectors, q)

    # Compute using individual rotations
    individual_results = np.array([rotate(v, q) for v in vectors])

    # Verify they match
    np.testing.assert_allclose(
        batch_result,
        individual_results,
        rtol=1e-5,
        atol=1e-5,
        err_msg="rotate_batch does not match individual rotate calls",
    )


def test_discretizer():
    k = 5

    discretizer = Discretizer(gymnasium.spaces.Box(-1, 1, shape=(1,)), k)
    step = 2.0 / (k - 1)

    for j in range(k):
        assert discretizer.discretize(discretizer.continuous(j)) == j, f"For index {j}"

    for x in np.arange(-1, 1, step):
        xhat = discretizer.continuous(discretizer.discretize(x))
        assert np.abs(xhat - x) < step, f"For value {x} vs {xhat}"


def test_path_cache_consistency():
    """Test that PathCache returns correct path indices for different starting nodes."""
    import heapq

    env = None
    max_paths = 50

    # Tracks known to have branches in STK
    tracks_with_branches = ["cocoa_temple", "zengarden", "snowtuxpeak"]

    try:
        env = STKRaceEnv(max_paths=max_paths)
        env.initialize(False)

        # Try to find a track with branches
        track_name = None
        for candidate in tracks_with_branches:
            env.current_track = candidate
            env.config = pystk2.RaceConfig(num_kart=1, track=candidate)
            env.warmup_race()
            if env.path_cache.has_branches:
                track_name = candidate
                break

        assert track_name is not None, (
            f"No track with branches found among {tracks_with_branches}"
        )

        track = env.track
        path_cache = env.path_cache

        # Verify path cache was created
        assert path_cache is not None

        # Test a few starting nodes
        for start_ix in [0, 10, 50, 100]:
            if start_ix >= path_cache.num_nodes:
                continue

            # Get cached result (with max_paths applied at query time)
            cached_paths = path_cache.get_path_indices(start_ix, max_paths)

            # Compute expected result using original algorithm
            path_distance = track.path_distance
            track_length = track.length
            successors = track.successors

            start_dist = path_distance[start_ix, 1]

            def get_distance(ix):
                dist = path_distance[ix, 1]
                return max(abs(dist - start_dist), track_length / 2)

            expected_paths = []
            path_heap = [(0.0, start_ix)]
            visited = set()

            for _ in range(max_paths):
                if not path_heap:
                    break
                _, ix = heapq.heappop(path_heap)
                if ix in visited:
                    continue
                visited.add(ix)
                expected_paths.append(ix)

                for succ_ix in successors[ix]:
                    if succ_ix not in visited:
                        heapq.heappush(path_heap, (get_distance(succ_ix), succ_ix))

            # Verify cached matches expected
            assert cached_paths == expected_paths, (
                f"PathCache mismatch for start_ix={start_ix}: "
                f"got {cached_paths[:10]}..., expected {expected_paths[:10]}..."
            )

    finally:
        if env is not None:
            env.close()
