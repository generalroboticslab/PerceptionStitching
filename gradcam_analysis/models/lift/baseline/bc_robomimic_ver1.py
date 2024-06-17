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
        default='robot0_eye_in_hand',
        help="The image for encoder 1. Can be frontview, agentview, sideview, robot0_eye_in_hand.",
    )

parser.add_argument(
        "--vision2",
        type=str,
        default='agentview',
        help="The image for encoder 2. Can be frontview, agentview, sideview, robot0_eye_in_hand.",
    )

parser.add_argument('--anchor_num', type=int, default=256, help='number of anchors')
parser.add_argument('--seed', type=int, default=101, help='random seed')

args = parser.parse_args()

print("seed:")
print(args.seed)

torch.manual_seed(args.seed)

class ImitationLearningDataset(Dataset):
    def __init__(self, file_path, vision1, vision2, mask_name=None):
        self.file = h5py.File(file_path, 'r')
        self.demos = [key for key in self.file['data'].keys() if "demo" in key]
        self.vision1 = vision1
        self.vision2 = vision2

        # Apply mask if provided
        if mask_name:
            mask = self.file['mask'][mask_name][:]
            self.demos = [self.demos[i] for i in range(len(self.demos)) if i < len(mask) and mask[i]]

        self.data_points = []

        for demo_name in self.demos:
            demo = self.file['data'][demo_name]
            num_steps = demo['actions'].shape[0]

            for step in range(num_steps):
                self.data_points.append((demo_name, step))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        demo_name, step = self.data_points[idx]
        demo = self.file['data'][demo_name]

        action = torch.tensor(demo['actions'][step], dtype=torch.float32)

        vision1_image = torch.tensor(demo['obs'][self.vision1 + '_image'][step],
                                   dtype=torch.float32).permute(2, 0, 1) / 255.0  # Adjust the shape to be (C, H, W) and norm from 255 to 1

        vision2_image = torch.tensor(demo['obs'][self.vision2 + '_image'][step],
                                   dtype=torch.float32).permute(2, 0, 1) / 255.0  # norm from 255 to 1

        # Concatenate images along the channel dimension
        images = torch.cat([vision1_image, vision2_image], dim=0)

        # Extract low dimensional observations
        eef_pos = torch.tensor(demo['obs']['robot0_eef_pos'][step], dtype=torch.float32)
        eef_quat = torch.tensor(demo['obs']['robot0_eef_quat'][step], dtype=torch.float32)
        gripper_qpos = torch.tensor(demo['obs']['robot0_gripper_qpos'][step], dtype=torch.float32)

        low_dim_obs = torch.cat([eef_pos, eef_quat, gripper_qpos], dim=0)

        return images, low_dim_obs, action


# Full dataset
dataset_name = 'datasets/lift/ph/FASRe_depth84.hdf5'

dataset = ImitationLearningDataset(dataset_name, vision1=args.vision1, vision2=args.vision2)

dataset_train = ImitationLearningDataset(dataset_name, vision1=args.vision1, vision2=args.vision2, mask_name='train')
dataset_valid = ImitationLearningDataset(dataset_name, vision1=args.vision1, vision2=args.vision2, mask_name='valid')

print(len(dataset))
print(len(dataset_train))
print(len(dataset_valid))

data_loader_train = DataLoader(dataset=dataset_train, sampler=None, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
data_loader_valid = DataLoader(dataset=dataset_valid, sampler=None, batch_size=16, shuffle=True, num_workers=2, drop_last=True)

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

# load the policy network
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
games_num = 20  # 20
total_reward = 0.
num_epochs = 100
VALIDATION_INTERVAL = 10
TEST_ROLLOUT_INTERVAL = 10  # 10
rollout_successes = 0

if args.log:
    writer = SummaryWriter('training_data/bc_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed))

if args.save_model:
    # full_model_path = "saved_models/lift/bc_robomimic_ver1_lr" + str(args.lr) + '_seed' + str(args.seed) + "_model.pt"
    model_path = "saved_models/lift/"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_file_name = 'bc_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_model.pt'

for epoch in range(num_epochs):
    # Training loop
    policy.train()
    running_loss = 0.0  # To accumulate the loss over batches
    num_batches = 0

    for images, low_dim_obs, actions in data_loader_train:

        # Send data to device
        images, low_dim_obs, actions = images.to(device), low_dim_obs.to(device), actions.to(device)
        # print(depths[:, 0:1])

        action_dist = policy.forward_train(
            low_dim_obs[:, 0:3],  # robot0_eef_pos
            low_dim_obs[:, 3:7],  # robot0_eef_quat
            low_dim_obs[:, 7:9],  # robot0_gripper_qpos
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

    # Validation loop
    if (epoch+1) % VALIDATION_INTERVAL == 0:
        policy.eval()
        validation_loss = 0
        with torch.no_grad():
            for images, low_dim_obs, actions in data_loader_valid:
                # Send data to device
                images, low_dim_obs, actions = images.to(device), low_dim_obs.to(device), actions.to(device)

                predicted_actions = policy(
                    low_dim_obs[:, 0:3],  # robot0_eef_pos
                    low_dim_obs[:, 3:7],  # robot0_eef_quat
                    low_dim_obs[:, 7:9],  # robot0_gripper_qpos
                    images[:, 0:3],  # robot0_eye_in_hand_image
                    images[:, 3:6],  # agentview_image
                )
                loss = eval_criterion(predicted_actions, actions)
                validation_loss += loss.item()
        avg_validation_loss = validation_loss / len(data_loader_valid)
        print(f"Epoch {epoch}, Validation Loss: {avg_validation_loss}")

        # Add to tensorboard - Validation
        if args.log:
            writer.add_scalar('validation_loss', avg_validation_loss, epoch)

    # Testing loop (rollout), and save policy network parameters
    if (epoch+1) % TEST_ROLLOUT_INTERVAL == 0:
        rollout_successes = 0
        policy.eval()
        with torch.no_grad():
            for game_i in range(games_num):
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
                    if done or success:
                        break
                    obs = deepcopy(next_obs)
            success_rate = rollout_successes / games_num
            print(f"Epoch {epoch}, Rollout Success Rate: {success_rate}")

            # Add to tensorboard - Rollout Success Rate
            if args.log:
                writer.add_scalar('rollout_success_rate', success_rate, epoch)
            # save the policy parameters
            if args.save_model:
                torch.save([policy.RGBView1ResnetEmbed.state_dict(), policy.RGBView3ResnetEmbed.state_dict(),
                            policy.Probot.state_dict()], model_path + model_file_name,
                           _use_new_zipfile_serialization=False)
