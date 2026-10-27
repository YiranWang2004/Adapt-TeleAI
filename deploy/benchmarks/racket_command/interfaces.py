"""Agent boundary: environments never reach into selectors or policy state."""

from typing import Any, Protocol

from deploy.racket_command import RacketCommand


class CommandAgent(Protocol):
  robot: Any  # MuJoCo plant exposing model/data for rendering only.
  dt: float
  translation: Any  # MuJoCo world -> command world, rendering only.
  metadata: dict
  accepted: bool
  decision: dict | None

  def reset(self) -> None: ...
  def advance(
    self, command: RacketCommand | None = None, on_physics_step=None
  ) -> None: ...
  def measure(self) -> dict: ...
  def lost_standing(self, sample: dict) -> bool: ...
