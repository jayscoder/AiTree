import unittest
from .config import *
from .utils import astar_find_path
import numpy as np

class TestMiniGridAStar(unittest.TestCase):
    def test_something(self):
        memory_obs = np.zeros((7, 7, 3), dtype=np.uint8)
        memory_obs[0, 0, 0] = Objects.agent.value
        for i in range(7):
            memory_obs[3, i, 0] = Objects.wall.value

        memory_obs[5, 5, 0] = Objects.goal.value

        path = astar_find_path(memory_obs, (0, 0), (5, 5))
        print(list(path or []))


if __name__ == '__main__':
    unittest.main()
