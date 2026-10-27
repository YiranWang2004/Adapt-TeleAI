"""World-frame task command shared by planners, selectors and benchmarks."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RacketCommand:
  position: np.ndarray
  velocity: np.ndarray
  normal: np.ndarray
  deadline: float
  diagnostic: dict
