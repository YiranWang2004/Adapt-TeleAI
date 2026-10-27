"""Wall-clock pacing only; never changes simulation steps or sensor timestamps."""

import math
import threading
import time

HELP = r"[ / - slower; ] / + faster; \ normal speed (0.0625x..8x)"


class PlaybackControl:
  def __init__(self, rate=1., *, clock=time.monotonic, sleep=time.sleep):
    if not math.isfinite(rate) or rate <= 0:
      raise ValueError("Playback rate must be finite and positive")
    self._rate = float(rate)
    self._clock, self._sleep = clock, sleep
    self._lock = threading.RLock()
    self._last_sim = None
    self._last_wall = clock()

  @property
  def rate(self):
    with self._lock:
      return self._rate

  def set_rate(self, rate):
    if not math.isfinite(rate) or rate <= 0:
      raise ValueError("Playback rate must be finite and positive")
    with self._lock:
      self._rate = float(rate)

  def key(self, key):
    with self._lock:
      if key in (ord("["), ord("-"), 333):  # GLFW keypad subtract
        self._rate = max(.0625, self._rate/2)
      elif key in (ord("]"), ord("="), ord("+"), 334):
        self._rate = min(8., self._rate*2)
      elif key == 92:  # GLFW_KEY_BACKSLASH; leave native numeric view keys alone.
        self._rate = 1.
      else:
        return False
      print(f"[MuJoCo] playback {self._rate:g}x; {HELP}", flush=True)
    return True

  def reset(self, sim_time=0.):
    """Rebase after a pause or episode reset, retaining the selected speed."""
    with self._lock:
      self._last_sim = float(sim_time)
      self._last_wall = self._clock()

  def wait(self, sim_time, *, frame_started=None):
    """Wait for one rendered frame, including its compute/render time.

    An overrun rebases the next deadline instead of running catch-up frames.
    Keyboard changes are checked during waits, including very slow playback.
    """
    with self._lock:
      delta = 0. if self._last_sim is None else max(0., sim_time-self._last_sim)
      self._last_sim = float(sim_time)
      started = self._last_wall if frame_started is None else frame_started
    while True:
      delay = started+delta/self.rate-self._clock()
      if delay <= 0:
        break
      self._sleep(min(delay, .02))
    with self._lock:
      self._last_wall = self._clock()
