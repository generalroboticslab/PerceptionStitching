import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import robosuite
from robosuite.controllers import load_controller_config
from nn_modules.resnet18_gmmmlp_view13rgb_model_ver1_low_dim_layer import PiNetwork
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
        "--log",
        action='store_true',
        help="Use the tensorboardX SummaryWriter to record the training curves"
    )

parser.add_argument(
        "--save_model",
        action='store_true',
        help="save the parameters of the policy network"
    )

parser.add_argument(
        "--vision1",
        type=str,
        default='eye_in_hand',
        help="The image for encoder 1. Can be front_view, agent_view, eye_in_hand.",
    )

parser.add_argument(
        "--vision2",
        type=str,
        default='agent_view',
        help="The image for encoder 2. Can be front_view, agent_view, eye_in_hand.",
    )

parser.add_argument('--anchor_num', type=int, default=256, help='number of anchors')
parser.add_argument('--seed', type=int, default=101, help='random seed')

args = parser.parse_args()

print("seed:")
print(args.seed)

torch.manual_seed(args.seed)

class ImitationLearningDataset(Dataset):
    def __init__(self, file_path, vision1, vision2):
        self.file = h5py.File(file_path, 'r')
        self.demos = [key for key in self.file.keys() if "demo" in key]
        self.vision1 = vision1
        self.vision2 = vision2

        self.data_points = []

        for demo_name in self.demos:
            demo = self.file[demo_name]
            num_steps = demo['actions'].shape[0]

            for step in range(num_steps):
                self.data_points.append((demo_name, step))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        demo_name, step = self.data_points[idx]
        demo = self.file[demo_name]

        action = torch.tensor(demo['actions'][step], dtype=torch.float32)

        vision1_image = torch.tensor(demo['obs'][self.vision1][step],
                                   dtype=torch.float32).permute(2, 0, 1) / 255.0  # Adjust the shape to be (C, H, W) and norm from 255 to 1

        vision2_image = torch.tensor(demo['obs'][self.vision2][step],
                                   dtype=torch.float32).permute(2, 0, 1) / 255.0  # norm from 255 to 1

        # Concatenate images along the channel dimension
        images = torch.cat([vision1_image, vision2_image], dim=0)

        # Extract low dimensional observations
        eef_pos = torch.tensor(demo['obs']['ee_pos'][step], dtype=torch.float32)  # 3
        eef_angle = torch.tensor(demo['obs']['ee_angle'][step], dtype=torch.float32)  # 3
        gripper_open = torch.tensor(demo['obs']['gripper_open'][step], dtype=torch.float32)  # 1
        gripper_open = gripper_open.unsqueeze(0)

        low_dim_obs = torch.cat([eef_pos, eef_angle, gripper_open], dim=0)

        return images, low_dim_obs, action


# Full dataset
dataset_name = 'datasets/lift/real/real_expert_lift_new.hdf5'

dataset = ImitationLearningDataset(dataset_name, vision1=args.vision1, vision2=args.vision2)

print(len(dataset))

data_loader = DataLoader(dataset=dataset, sampler=None, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

# load the policy network
input_shape = [512, 3, 3]
image_latent_dim = args.anchor_num
action_dim = 7
low_dim_input_dim = 3 + 3 + 1  # robot0_eef_pos + robot0_eef_angle + robot0_gripper_open
mlp_hidden_dims = [1024, 1024]
device = torch.device(args.device)

policy = PiNetwork(input_shape, image_latent_dim, action_dim, low_dim_input_dim, mlp_hidden_dims)
policy.to(device)
policy.float()

# start the training process

# Initialize the optimizer and validation loss criterion
optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=0.0)
eval_criterion = nn.MSELoss()

num_epochs = 100
TEST_ROLLOUT_INTERVAL = 10  # 10
rollout_successes = 0

if args.log:
    writer = SummaryWriter('training_data/real_lift/real_bc_robomimic_ver1_low_dim_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed))

if args.save_model:
    # full_model_path = "saved_models/lift/bc_robomimic_ver1_lr" + str(args.lr) + '_seed' + str(args.seed) + "_model.pt"
    model_path = "saved_models/real_lift/"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_file_name = 'real_bc_robomimic_ver1_low_dim_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_model.pt'

for epoch in range(num_epochs):
    # Training loop
    policy.train()
    running_loss = 0.0  # To accumulate the loss over batches
    num_batches = 0

    for images, low_dim_obs, actions in data_loader:

        # Send data to device
        images, low_dim_obs, actions = images.to(device), low_dim_obs.to(device), actions.to(device)
        # print(depths[:, 0:1])

        action_dist = policy.forward_train(
            low_dim_obs[:, 0:3],  # robot0_eef_pos
            low_dim_obs[:, 3:6],  # robot0_eef_quat
            low_dim_obs[:, 6:7],  # robot0_gripper_qpos
            images[:, 0:3],  # robot0_eye_in_hand_image
            images[:, 3:6],  # agentview_image
        )
        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(action_dist.batch_shape) == 1
        log_probs = action_dist.log_prob(actions)
        # loss is just negative log-likelihood of action targets
        loss = -log_probs.mean()

        # backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
    avg_training_loss = running_loss / num_batches

    print(f"Training epoch {epoch} - Average Training Loss: {avg_training_loss:.4f}")

    # Add to tensorboard - Training
    if args.log:
        writer.add_scalar('average_training_loss', avg_training_loss, epoch)

    # No testing(rollout), but save policy network parameters
    if (epoch+1) % TEST_ROLLOUT_INTERVAL == 0:
            # save the policy parameters
            if args.save_model:
                print("save model")
                torch.save([policy.RGBView1ResnetEmbed.state_dict(), policy.RGBView3ResnetEmbed.state_dict(),
                            policy.Probot.state_dict()], model_path + model_file_name,
                           _use_new_zipfile_serialization=False)
