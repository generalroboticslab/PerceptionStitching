import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import robosuite
from robosuite.controllers import load_controller_config
from nn_modules.resnet18_gmmmlp_view13rgb_model_ver1 import PiNetwork
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import argparse
from tensorboardX import SummaryWriter
from collections import OrderedDict
import os

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
        default='vision1agentview_vision2sideview',
        help="record the two visions of the policy",
    )

parser.add_argument(
        "--mlp_suffix",
        type=str,
        default='vision1robot0_eye_in_hand_vision2sideview',
        help="record the two visions of the policy",
    )
parser.add_argument('--anchor_num', type=int, default=256, help='number of anchors')
parser.add_argument('--games_num', type=int, default=50, help='number of games for testing')

parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--view1_seed', type=int, default=101, help='random seed')
parser.add_argument('--view3_seed', type=int, default=101, help='random seed')
parser.add_argument('--mlp_seed', type=int, default=101, help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)

# Full dataset
dataset_name = 'datasets/lift/ph/image_v141.hdf5'

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
device = torch.device(args.device)

policy = PiNetwork(input_shape, image_latent_dim, action_dim, low_dim_input_dim, mlp_hidden_dims)
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

vision1_encoder_path = 'saved_models/lift/bc_robomimic_ver1_' + args.vision1_suffix + '_anchors' + str(args.anchor_num) + '_lr0.0001_seed101_model.pt'
vision2_encoder_path = 'saved_models/lift/bc_robomimic_ver1_' + args.vision2_suffix + '_anchors' + str(args.anchor_num) + '_lr0.0001_seed102_model.pt'
gmm_mlp_path = 'saved_models/lift/bc_robomimic_ver1_' + args.mlp_suffix + '_anchors' + str(args.anchor_num) + '_lr0.0001_seed102_model.pt'

print(vision1_encoder_path)
print(vision2_encoder_path)
print(gmm_mlp_path)

data1 = torch.load(vision1_encoder_path, map_location=device)  # data1 for task vision1_encoder
data2 = torch.load(vision2_encoder_path, map_location=device)  # data2 for vision2_encoder
data3 = torch.load(gmm_mlp_path, map_location=device)  # data3 for gmm_mlp

policy.RGBView1ResnetEmbed.load_state_dict(data1[0])
policy.RGBView3ResnetEmbed.load_state_dict(data2[1])
policy.Probot.load_state_dict(data3[2])

policy.eval()
with torch.no_grad():
    for game_i in range(games_num):
        print(game_i)
        obs = env.reset()
        for step_i in range(horizon):
            tensor_obs = {key: torch.tensor(value.copy(), dtype=torch.float32, device=device) for key, value in obs.items()}
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
    print(f"Rollout Success Rate: {success_rate}")
