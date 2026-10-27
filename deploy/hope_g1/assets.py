"""Use the G1 checkpoint's own fast-URDF collision shapes in the local model."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from deploy.g1_pingpong.core import quaternion_multiply, rotation


def urdf_pose(origin):
  if origin is None:
    return np.zeros(3), np.array([1.0, 0, 0, 0])
  position = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
  roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
  quaternion = quaternion_multiply(
    quaternion_multiply(
      [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
      [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0],
    ),
    [np.cos(roll / 2), np.sin(roll / 2), 0, 0],
  )
  return position, quaternion


def training_collision_spec(base_factory, urdf, self_collisions):
  """Preserve the existing joint/inertial/visual asset; replace collision only.

  Fixed URDF links (head, mount, paddle) are expressed in their nearest MuJoCo
  ancestor body. No mass is inferred from the newly added collision geometry.
  """
  spec = base_factory()
  root = ET.parse(urdf).getroot()
  parents = {j.find("child").get("link"): j for j in root.findall("joint")}
  bodies = {body.name: body for body in spec.bodies}
  meshes = {mesh.name for mesh in spec.meshes}
  for geom in list(spec.geoms):
    if geom.name.endswith("_collision"):
      spec.delete(geom)
  count = 0
  for link in root.findall("link"):
    for collision in link.findall("collision"):
      body = link.get("name")
      pos, quat = urdf_pose(collision.find("origin"))
      while body not in bodies:
        joint = parents[body]
        p, q = urdf_pose(joint.find("origin"))
        pos, quat = p + rotation(q) @ pos, quaternion_multiply(q, quat)
        body = joint.find("parent").get("link")
      geometry = list(collision.find("geometry"))[0]
      kind = geometry.tag
      args = {}
      if kind == "box":
        args.update(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=np.fromstring(geometry.get("size"), sep=" ") / 2,
        )
      elif kind == "sphere":
        args.update(
          type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[float(geometry.get("radius")), 0, 0]
        )
      elif kind == "mesh":
        name = Path(geometry.get("filename")).stem
        if name not in meshes:
          raise ValueError(f"Training collision mesh is unavailable locally: {name}")
        if geometry.get("scale", "1 1 1") != "1 1 1":
          raise ValueError("Unsupported non-unit collision mesh scale")
        args.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
      else:
        raise ValueError(f"Unsupported training collision geometry: {kind}")
      bodies[body].add_geom(
        name=f"hope_training_shape_{count}",
        pos=pos,
        quat=quat,
        contype=1 if self_collisions else 0,
        conaffinity=1,
        condim=3,
        friction=[1.0, 0.005, 0.0001],
        mass=0.0,
        rgba=[0.5, 0.5, 0.5, 0],
        **args,
      )
      count += 1
  return spec
