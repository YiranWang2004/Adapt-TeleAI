"""Actor-only CPU inference for the original HOPE RSL-RL checkpoint format."""

import numpy as np
import torch
from torch import nn


class HopePolicy:
  input_dim = 175
  output_dim = 29

  def __init__(self, checkpoint, agent):
    cfg = agent["policy"]
    if cfg["class_name"] != "ActorCritic" or cfg["activation"] != "elu":
      raise ValueError("Expected HOPE feed-forward ActorCritic with ELU")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = {
      k.removeprefix("actor."): v
      for k, v in saved["model_state_dict"].items()
      if k.startswith("actor.")
    }
    dims = [self.input_dim, *cfg["actor_hidden_dims"], self.output_dim]
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
      layers.append(nn.Linear(a, b))
      if i < len(dims) - 2:
        layers.append(nn.ELU())
    self.actor = nn.Sequential(*layers).eval()
    self.actor.load_state_dict(weights, strict=True)
    if not agent.get("empirical_normalization"):
      raise ValueError("This HOPE adapter requires saved empirical normalization")
    norm = saved["obs_norm_state_dict"]
    self.mean, self.std = norm["_mean"].float(), norm["_std"].float()
    if (
      self.mean.shape != (1, self.input_dim)
      or self.std.shape != self.mean.shape
      or not torch.isfinite(self.mean).all()
      or not torch.isfinite(self.std).all()
      or torch.any(self.std < 0)
      or any(not torch.isfinite(w).all() for w in self.actor.parameters())
    ):
      raise ValueError("Invalid actor/normalizer tensors")
    # rsl_rl.modules.normalizer.EmpiricalNormalization default in HOPE's env.
    self.eps = 1e-2
    self.iteration = int(saved["iter"])
    self.clip = agent.get("clip_actions")
    if self.clip is not None and (not np.isfinite(self.clip) or self.clip <= 0):
      raise ValueError("Invalid runner action clipping")
    self(np.zeros(self.input_dim, dtype=np.float32))

  def __call__(self, observation):
    x = np.asarray(observation, dtype=np.float32)
    if x.shape != (self.input_dim,) or not np.isfinite(x).all():
      raise ValueError("HOPE requires a finite 175-dimensional observation")
    with torch.inference_mode():
      normalized = (torch.from_numpy(x[None]) - self.mean) / (self.std + self.eps)
      action = self.actor(normalized).numpy()[0].copy()
    if self.clip is not None:
      action = np.clip(action, -self.clip, self.clip)
    if action.shape != (29,) or not np.isfinite(action).all():
      raise ValueError("Invalid HOPE actor output")
    return action
