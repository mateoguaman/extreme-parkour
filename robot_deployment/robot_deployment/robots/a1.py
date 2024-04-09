"""Base class for all robots."""
import numpy as np
from typing import Sequence
from typing import Tuple
from pybullet_utils.bullet_client import BulletClient

from configs.definitions import DeploymentConfig
from robot_deployment.robots.motors import MotorControlMode
from robot_deployment.robots.robot import Robot

class A1(Robot):
  """A1 Robot."""
  def __init__(
      self,
      pybullet_client: BulletClient,
      sim_conf: DeploymentConfig,
      urdf_path: str = "a1_extreme_parkour.urdf",  ## Changed by Mateo
      base_joint_names: Tuple[str, ...] = (),  ## Unsure why this is empty
      foot_joint_names: Tuple[str, ...] = (  ## Also changed by Mateo
          "FR_foot_fixed",
          "FL_foot_fixed",
          "RR_foot_fixed",
          "RL_foot_fixed"
      ),
      motor_control_mode: MotorControlMode = MotorControlMode.POSITION,
      mpc_body_height: float = 0.3,
      mpc_body_mass: float = 110 / 9.8,
      mpc_body_inertia: Tuple[float] = np.array(
          (0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 10.,
  ) -> None:
    """Constructs an A1 robot and resets it to the initial states.
        Initializes a tuple with a single MotorGroup containing 12 MotoroModels.
        Each MotorModel is by default configured for the parameters of the A1.
        """
    motor_names = ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                   "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                   "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                   "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint")
    self._mpc_body_height = mpc_body_height
    self._mpc_body_mass = mpc_body_mass
    self._mpc_body_inertia = mpc_body_inertia

    super().__init__(
      urdf_path,
      pybullet_client,
      sim_conf,
      motor_names,
      motor_control_mode,
      base_joint_names,
      foot_joint_names
    )

  @property
  def mpc_body_height(self):
    return self._mpc_body_height

  @mpc_body_height.setter
  def mpc_body_height(self, mpc_body_height: float):
    self._mpc_body_height = mpc_body_height

  @property
  def mpc_body_mass(self):
    return self._mpc_body_mass

  @mpc_body_mass.setter
  def mpc_body_mass(self, mpc_body_mass: float):
    self._mpc_body_mass = mpc_body_mass

  @property
  def mpc_body_inertia(self):
    return self._mpc_body_inertia

  @mpc_body_inertia.setter
  def mpc_body_inertia(self, mpc_body_inertia: Sequence[float]):
    self._mpc_body_inertia = mpc_body_inertia

  @property
  def swing_reference_positions(self):
    return (
        (0.17, -0.135, 0),
        (0.17, 0.13, 0),
        (-0.195, -0.135, 0),
        (-0.195, 0.13, 0),
    )

  @property
  def num_motors(self):
    return 12
