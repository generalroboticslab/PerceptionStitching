import sys
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import robosuite
from robosuite.controllers import load_controller_config
from nn_modules.resnet18_gmmmlp_view13rgb_rel_model_low_dim_linear_sum import PiNetwork
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import argparse
from tensorboardX import SummaryWriter
from collections import OrderedDict
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

# Try to reproduce the bc.json for Lift/ph training
# only rgb images, no depth images
# no random crop of the image input

parser = argparse.ArgumentParser()

parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate",
    )

parser.add_argument(
        "--device",
        type=str,
        default='cuda:2',
        help="the device for training",
    )

parser.add_argument(
        "--vision1",
        type=str,
        default='robot0_eye_in_hand',
        help="The image for encoder 1. Can be frontview, agentview, sideview, robot0_eye_in_hand.",
    )

parser.add_argument(
        "--vision2",
        type=str,
        default='agentview',
        help="The image for encoder 2. Can be frontview, agentview, sideview, robot0_eye_in_hand.",
    )

parser.add_argument(
        "--vision1_suffix",
        type=str,
        default='vision1robot0_eye_in_hand_vision2frontview',
        help="record the two visions of the policy",
    )

parser.add_argument(
        "--vision2_suffix",
        type=str,
        default='vision1robot0_eye_in_hand_vision2sideview',
        help="record the two visions of the policy",
    )

parser.add_argument(
        "--mlp_suffix",
        type=str,
        default='vision1robot0_eye_in_hand_vision2frontview',
        help="record the two visions of the policy",
    )

parser.add_argument('--games_num', type=int, default=50, help='number of games for testing')
parser.add_argument('--anchor_num', type=int, default=256, help='number of anchors')
parser.add_argument('--seed', type=int, default=101, help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)

# Full dataset
dataset_name = 'datasets/lift/ph/FASRe_depth84.hdf5'

# load anchors
device = torch.device(args.device)
if args.vision1 == 'agentview':
    vision1_anchors = np.load('hdf5_image/lift/' + args.vision1 + '_' +str(args.anchor_num) + 'anchor_images.npy')
else:
    vision1_anchors = np.load('hdf5_image/lift/' + args.vision1 + '_' +str(args.anchor_num) + 'anchor_images_from_agentview_idx.npy')

if args.vision2 == 'agentview':
    vision2_anchors = np.load('hdf5_image/lift/' + args.vision2 + '_' +str(args.anchor_num) + 'anchor_images.npy')
else:
    vision2_anchors = np.load('hdf5_image/lift/' + args.vision2 + '_' +str(args.anchor_num) + 'anchor_images_from_agentview_idx.npy')

vision1_anchors_tensor = torch.tensor(vision1_anchors, dtype=torch.float32).to(device).permute(0, 3, 1, 2) / 255.0
vision2_anchors_tensor = torch.tensor(vision2_anchors, dtype=torch.float32).to(device).permute(0, 3, 1, 2) / 255.0

# create environment from dataset
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_name)
env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[args.vision1, args.vision2],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
        use_depth_obs=False,
    )

# initialize the policy network
input_shape = [512, 3, 3]
image_latent_dim = args.anchor_num
action_dim = 7
low_dim_input_dim = 3 + 4 + 2  # robot0_eef_pos + robot0_eef_quat + robot0_gripper_qpos
mlp_hidden_dims = [1024, 1024]

policy = PiNetwork(input_shape, vision1_anchors_tensor, vision2_anchors_tensor, image_latent_dim, action_dim, low_dim_input_dim, mlp_hidden_dims)
policy.to(device)
policy.float()

# start the training process

# Initialize the optimizer and validation loss criterion
optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=0.0)
eval_criterion = nn.MSELoss()

horizon = 400
games_num = args.games_num
rollout_successes = 0

# test load and rollout
robot0_eye_in_hand_image_encoder_path = 'your_model_path'
agentview_image_encoder_path = 'your_model_path'
gmm_mlp_path = 'your_model_path'

print(robot0_eye_in_hand_image_encoder_path)
print(agentview_image_encoder_path)
print(gmm_mlp_path)

data1 = torch.load(robot0_eye_in_hand_image_encoder_path, map_location=device)  # data1 for task robot0_eye_in_hand_image_encoder
data2 = torch.load(agentview_image_encoder_path, map_location=device)  # data2 for agentview_image_encoder
data3 = torch.load(gmm_mlp_path, map_location=device)  # data3 for gmm_mlp

policy.RGBView1ResnetEmbed.load_state_dict(data1[0])
policy.RGBView3ResnetEmbed.load_state_dict(data2[1])
policy.Probot.load_state_dict(data3[2])

policy.eval()

repeat_num = 3
success_rate_list = []

with torch.no_grad():
    for i in range(repeat_num):
        rollout_successes = 0
        for game_i in range(games_num):
            print(game_i)
            obs = env.reset()
            for step_i in range(horizon):
                tensor_obs = {key: torch.tensor(value.copy(), dtype=torch.float32, device=device) for key, value in
                              obs.items()}
                pi = policy(
                    tensor_obs['robot0_eef_pos'].unsqueeze(0),
                    tensor_obs['robot0_eef_quat'].unsqueeze(0),
                    tensor_obs['robot0_gripper_qpos'].unsqueeze(0),
                    tensor_obs[args.vision1 + '_image'].permute(2, 0, 1).unsqueeze(0) / 255.0,  # norm from 255 to 1
                    tensor_obs[args.vision2 + '_image'].permute(2, 0, 1).unsqueeze(0) / 255.0,  # norm from 255 to 1
                )
                act = pi.cpu().squeeze().numpy()
                next_obs, r, done, _ = env.step(act)
                success = env.is_success()["task"]
                if success:
                    rollout_successes += 1
                    print("success")
                if done or success:
                    break
                obs = deepcopy(next_obs)
        success_rate = rollout_successes / games_num
        success_rate_list.append(success_rate)
        print(f"Rollout Success Rate: {success_rate}")

success_rate_list = np.array(success_rate_list)
mean_success_rate = np.mean(success_rate_list)
std_success_rate = np.std(success_rate_list)

print(f"Mean Success Rate: {mean_success_rate:.6f}, Std Success Rate: {std_success_rate:.6f}")
