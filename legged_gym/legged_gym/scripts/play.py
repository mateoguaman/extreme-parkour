# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.2,
                                    "parkour_hurdle": 0.2,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, 
                                    "demo": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]  ## Mateo: Unsure what this depth.angle thing means or does here
    env_cfg.noise.add_noise = True  ## Mateo: Unsure why we have add_noise in play
    env_cfg.domain_rand.randomize_friction = True  ## Mateo: Unsure why we randomize friction in play
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()  ## Mateo: TODO: Check dimensionality of observations. Answer: It's a [16, 753] dimensional array, env_cfg.env.num_envs is 16, so [num_envs, 753]. 

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    if args.use_jit:  ## Mateo: For now I think we can assume we use a jitted policy, TODO: verify if this is True
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device) ## Mateo: TODO: Check what type of network this is
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)  ## Mateo: Seems like it doesn't really load a policy unless it's jitted, which is odd. Maybe this branch of the code is incomplete.  ## Mateo: TODO: Check what type of network this is
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)  ## Mateo: Seems like there is an estimator that gets loaded 
    if env.cfg.depth.use_camera:  ## Mateo: This should be true in our case
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)  ## Mateo: TODO: Check what type of network this is 

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None  ## MEGA TODO: Check what this dimensionality is. I don't like that this is stored in infos instead of observations, and we should change that, but that might require serializing the depth image to be 1D so it can be concatenated with other states.

    for i in range(10*int(env.max_episode_length)):  ## Mateo: Why is there a 10*max_episode_length here. Is it because they consider histories of 10 observations at a time? Unclear.
        if args.use_jit:  ## ## Mateo: TODO: Check that this is true
            if env.cfg.depth.use_camera:  ## Mateo: TODO: Check that this is true
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)  ## Mateo: TODO: The depth latent is always an array of ones here. This makes no sense. Check what this is supposed to be
                    actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)  ## Mateo: TODO: Check what type of network this is, and what are the four inputs that it allows. Also, why does it return the depth latent? 
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)  ## Mateo: This is saying that if there is no depth image in the observation, you pass a 2D array of size (58, 87) into the network. Notably, the second argument is False instead of True, and the depth_latent that doesn't seem like it's actually set anywhere before this, so SHOULD break.
                    actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)  ## Mateo: TODO: Check what happens if you don't pass the camera. Seems like you remove some of the observations in the middle
                actions = policy(obs_jit)  ## Mateo: Note that policy and policy_jit seem to be different functions. policy() of them only takes one parameter as input, while policy_jit seems to take 4. This suggests that policy may only be used during training, and policy_jit may be used during play/deployment
        else:  ## Mateo: If not using jit, which I'm not sure when this is true but seems to actually correspond to different logic
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:  ## Mateo: So if not jit, using camera, and a depth image exists
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()  ## Mateo: Seems to be copying the proprioception info into obs_student
                    obs_student[:, 6:8] = 0  ## Mateo: They're manually zeroing out 1 of the fields TODO: Check which. This is so hacky. Seems to be yaw based on line 157.
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)  ## Mateo: They extract a depth latent and yaw by passing a depth image and the extracted proprioception observation into the depth encoder network
                    depth_latent = depth_latent_and_yaw[:, :-2] ## Mateo: Split up depth_latent and yaw
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw  ## Mateo: Similar as in line 142, they don't have yaw defined outside the if statement above, so this should only work if there IS a valid depth image. I'm assuming that if not, they want to re=use the value from the last time you got a valid image, so there may be a condition above that only runs if there's at least one depth image.
                    
            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):  ## Mateo: TODO: When is this true? Probably after the distillation step, so during play and deployment. Check that this is true during play.
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)  ## Mateo: Get actions from the depth_actor rather than from the "policy."  
            else:
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
