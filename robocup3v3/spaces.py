"""Gym-compatible spaces with a tiny fallback for dependency-light tests."""

from __future__ import annotations

import numpy as np

try:  # Prefer the exact legacy dependency used by the supplied on-policy code.
    from gym.spaces import Box, Discrete, MultiDiscrete  # type: ignore
except ImportError:
    try:
        from gymnasium.spaces import Box, Discrete, MultiDiscrete  # type: ignore
    except ImportError:
        class Box:  # pragma: no cover - exercised on the minimal server image
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.dtype = np.dtype(dtype)
                if shape is not None:
                    self.low = np.full(shape, low, dtype=self.dtype)
                    self.high = np.full(shape, high, dtype=self.dtype)
                else:
                    self.low = np.asarray(low, dtype=self.dtype)
                    self.high = np.asarray(high, dtype=self.dtype)
                self.shape = self.low.shape

            def sample(self):
                return np.random.uniform(self.low, self.high).astype(self.dtype)

            def contains(self, value):
                value = np.asarray(value)
                return value.shape == self.shape and bool(
                    np.all(value >= self.low) and np.all(value <= self.high)
                )

        class Discrete:
            def __init__(self, n):
                self.n = int(n)
                self.shape = ()

            def sample(self):
                return int(np.random.randint(self.n))

            def contains(self, value):
                return 0 <= int(value) < self.n

        class MultiDiscrete:
            def __init__(self, nvec):
                self.nvec = np.asarray(nvec, dtype=np.int64)
                # Old MAPPO detects this class by name and reads .high/.low.
                self.low = np.zeros_like(self.nvec)
                self.high = self.nvec - 1
                self.shape = self.nvec.shape

            def sample(self):
                return np.asarray(
                    [np.random.randint(int(n)) for n in self.nvec], dtype=np.int64
                )

            def contains(self, value):
                value = np.asarray(value)
                return value.shape == self.shape and bool(
                    np.all(value >= 0) and np.all(value < self.nvec)
                )

