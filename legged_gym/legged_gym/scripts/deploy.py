import isaacgym
import logging
import os.path as osp
import time

from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
# from pydantic import TypeAdapter

from configs.hydra import ExperimentHydraConfig
from configs.definitions import (EnvConfig, TaskConfig, TrainConfig, ObservationConfig,
                                 SimConfig, RunnerConfig, TerrainConfig)
from configs.definitions import DeploymentConfig
from configs.overrides.domain_rand import NoDomainRandConfig
from configs.overrides.noise import NoNoiseConfig
# from legged_gym.envs.a1 import A1
# from legged_gym.utils.observation_buffer import ObservationBuffer
# from legged_gym.utils.helpers import (export_policy_as_jit, get_load_path, get_latest_experiment_path,
#                                       empty_cfg, from_repo_root, save_config_as_yaml)
# from rsl_rl.runners import OnPolicyRunner
from robot_deployment.envs.locomotion_gym_env import LocomotionGymEnv
import torch

# from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import get_args, class_to_dict

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.a1.a1_parkour_config import A1ParkourCfg, A1ParkourCfgPPO
from rsl_rl.runners.deploy_on_policy_runner import DeployOnPolicyRunner

# OmegaConf.register_new_resolver("not", lambda b: not b)
# OmegaConf.register_new_resolver("compute_timestep", lambda dt, decimation, action_repeat: dt * decimation / action_repeat)

INIT_JOINT_ANGLES = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

@dataclass
class DeployScriptConfig:
    # checkpoint_root: str = from_repo_root("../experiment_logs/train")
    # logging_root: str = from_repo_root("../experiment_logs")
    # export_policy: bool = True
    use_joystick: bool = True
    # episode_length_s: float = 200.
    # checkpoint: int = -1
    device: str = "cpu"
    use_real_robot: bool = False

    # hydra: ExperimentHydraConfig = ExperimentHydraConfig()

    # task: TaskConfig = empty_cfg(TaskConfig)(
    #     env = empty_cfg(EnvConfig)(
    #         num_envs = "${num_envs}"
    #     ),
    #     observation = empty_cfg(ObservationConfig)(
    #         get_commands_from_joystick = "${use_joystick}"
    #     ),
    #     sim = empty_cfg(SimConfig)(
    #         device = "${device}",
    #         use_gpu_pipeline = "${evaluate_use_gpu: ${task.sim.device}}",
    #         headless = True, # meaning Isaac is headless, but not the target for deployment 
    #         physx = empty_cfg(SimConfig.PhysxConfig)(
    #             use_gpu = "${evaluate_use_gpu: ${task.sim.device}}"
    #         )
    #     ),
    #     terrain = empty_cfg(TerrainConfig)(
    #         curriculum = False
    #     ),
    #     noise = NoNoiseConfig(),
    #     domain_rand = NoDomainRandConfig()
    # ) 
    # train: TrainConfig = empty_cfg(TrainConfig)(
    #     device = "${device}",
    #     log_dir = "${hydra:runtime.output_dir}",
    #     runner = empty_cfg(RunnerConfig)(
    #         checkpoint="${checkpoint}"
    #     )
    # )
    deployment: DeploymentConfig = DeploymentConfig(
        use_real_robot="${use_real_robot}",
        get_commands_from_joystick="${use_joystick}",
        render=DeploymentConfig.RenderConfig(
            show_gui="${not: ${use_real_robot}}"
        ),
        # timestep="${compute_timestep: ${task.sim.dt}, ${task.control.decimation}, ${deployment.action_repeat}}",
        timestep=0.002, #TODO: should also try 0.0004 w dt=0.001. their dt 0.005, decimation=4, unknown action_repeat (we assume it's 10 form our code), and the formuoli we use is dt*decimation/action_repeat
        init_position=(0.0, 0.0, 0.42),
        init_joint_angles=INIT_JOINT_ANGLES,
        stiffness=dict(joint=40.),
        damping=dict(joint=1.),
        action_scale=0.25
    )

cs = ConfigStore.instance()
cs.store(name="config", node=DeployScriptConfig)

@hydra.main(version_base=None, config_name="config")
def main(cfg: DeployScriptConfig):
    # experiment_path = get_latest_experiment_path(cfg.checkpoint_root)
    # latest_config_filepath = osp.join(experiment_path, "resolved_config.yaml")
    # log.info(f"1. Deserializing policy config from: {osp.abspath(latest_config_filepath)}")
    # loaded_cfg = OmegaConf.load(latest_config_filepath)

    # log.info("2. Merging loaded config, defaults and current top-level config.")
    # del(loaded_cfg.hydra) # Remove unpopulated hydra configuration key from dictionary
    # default_cfg = {"task": TaskConfig(), "train": TrainConfig()}  # default behaviour as defined in "configs/definitions.py"
    # merged_cfg = OmegaConf.merge(
    #     default_cfg,  # loads default values at the end if it's not specified anywhere else
    #     loaded_cfg,   # loads values from the previous experiment if not specified in the top-level config
    #     cfg           # highest priority, loads from the top-level config dataclass above
    # )
    # # Resolves the config (replaces all "interpolations" - references in the config that need to be resolved to constant values)
    # # and turns it to a dictionary (instead of DictConfig in OmegaConf). Throws an error if there are still missing values.
    # merged_cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
    # # Creates a new DeployScriptConfig object (with type-checking and optional validation) using Pydantic.
    # # The merged config file (DictConfig as given by OmegaConf) has to be recursively turned to a dict for Pydantic to use it.
    # cfg = TypeAdapter(DeployScriptConfig).validate_python(merged_cfg_dict)
    # # cfg = merged_cfg
    # # Alternatively, you should be able to use "from pydantic.dataclasses import dataclass" and replace the above line with
    # # cfg = PlayScriptConfig(**merged_cfg_dict)
    # log.info(f"3. Printing merged cfg.")
    # print(OmegaConf.to_yaml(cfg))
    # save_config_as_yaml(cfg)

    log.info(f"4. Preparing Isaac environment and runner.")
    # task_cfg = cfg.task
    # isaac_env: A1 = hydra.utils.instantiate(task_cfg)

    ## Mateo: THE MESS STARTS HERE
    # env, env_cfg = task_registry.make_env(name=args.task, args=args)
    args = get_args()
    import os
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')
    args.exptid = "001-00"
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    # import ipdb;ipdb.set_trace()
    env_cfg = A1ParkourCfg()
    train_cfg = A1ParkourCfgPPO()
    train_cfg_dict = class_to_dict(train_cfg)
    runner = DeployOnPolicyRunner(env_cfg, 
                            train_cfg_dict, 
                            "./", 
                            init_wandb=True,
                            device=args.rl_device)
    # ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env_cfg=env_cfg, name=args.task, args=args)
    # sim_params = 
    # env = LeggedRobot(A1ParkourCfg, sim_params=sim_params,
    #                         physics_engine=args.physics_engine,
    #                         sim_device=args.sim_device,
    #                         headless=args.headless))
    import ipdb;ipdb.set_trace()

    
    # runner: OnPolicyRunner = hydra.utils.instantiate(cfg.train, env=isaac_env, _recursive_=False)
    # runner = OnPolicyRunner()

    experiment_path = get_latest_experiment_path(cfg.checkpoint_root)
    resume_path = get_load_path(experiment_path, checkpoint=cfg.train.runner.checkpoint)
    log.info(f"5. Loading policy checkpoint from: {resume_path}.")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=isaac_env.device)

    if cfg.export_policy:
        export_policy_as_jit(runner.alg.actor_critic, cfg.checkpoint_root)
        log.info(f"Exported policy as jit script to: {cfg.checkpoint_root}")

    log.info(f"6. Instantiating robot deployment environment.")
    # create robot environment (either in PyBullet or real world)
    deploy_env = LocomotionGymEnv(
        cfg.deployment,
        cfg.task.observation.sensors,
        cfg.task.normalization.obs_scales,
        cfg.task.commands.ranges
    )

    obs, info = deploy_env.reset()
    for _ in range(1):
        obs, *_, info = deploy_env.step(deploy_env.default_motor_angles)

    obs_buf = ObservationBuffer(1, isaac_env.num_obs, task_cfg.observation.history_steps, runner.device)

    all_actions = []
    all_infos = None

    log.info(f"7. Running the inference loop.")
    for t in range(int(cfg.episode_length_s / deploy_env.robot.control_timestep)):
        # Form observation for policy.
        obs = torch.tensor(obs, device=runner.device).float()
        if t == 0:
            obs_buf.reset([0], obs)
            all_infos = {k: [v.copy()] for k, v in info.items()}
        else:
            obs_buf.insert(obs)
            for k, v in info.items():
                all_infos[k].append(v.copy())

        policy_obs = obs_buf.get_obs_vec(range(task_cfg.observation.history_steps))

        # Evaluate policy and act.
        actions = policy(policy_obs.detach()).detach().cpu().numpy().squeeze()
        actions = task_cfg.control.action_scale*actions + deploy_env.default_motor_angles
        all_actions.append(actions)
        obs, _, terminated, _, info = deploy_env.step(actions)

        if terminated:
            log.warning("Unsafe, terminating!")
            break

    log.info("8. Exit Cleanly")
    isaac_env.exit()
    # TODO: check if target simulator (pybullet or other) has exit logic


if __name__ == '__main__':
    log = logging.getLogger(__name__)
    main()
