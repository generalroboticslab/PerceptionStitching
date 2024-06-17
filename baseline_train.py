import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import robosuite
from robosuite.controllers import load_controller_config
from nn_modules.resnet18_LSTMgmm_view13rgb_model_ver1 import PiNetwork
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import argparse
from tensorboardX import SummaryWriter
from collections import OrderedDict
import os
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from datetime import datetime

# Try to reproduce the bc_rnn.json for can, square, tool_hang/ph training
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
        help="The image for encoder 1. Can be frontview, agentview, sideview, robot0_eye_in_hand, robot0_robotview.",
    )

parser.add_argument(
        "--vision2",
        type=str,
        default='agentview',
        help="The image for encoder 2. Can be frontview, agentview, sideview, robot0_eye_in_hand, robot0_robotview.",
    )

parser.add_argument(
        "--task",
        type=str,
        default='can',
        help="can, suqare, tool_hang",
    )

parser.add_argument(
        "--dataset_name",
        type=str,
        default='FARrRe_depth84',
        help="FARrRe_depth84, FASRe_depth84.hdf5, FASRe_depth240",
    )

parser.add_argument(
        "--process1",
        action='store_true',
        help="process on vision1"
    )

parser.add_argument(
        "--process2",
        action='store_true',
        help="process on vision2"
    )

parser.add_argument(
        "--effect",
        type=str,
        default="mask",
        help="mask zoomin blur noise fisheye, mask default size is 14",
    )

parser.add_argument('--square_size', type=int, default=14, help='size of the square mask')
parser.add_argument('--anchor_num', type=int, default=256, help='number of anchors')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--horizon', type=int, default=400, help='horizon of a game, 700 for tool_hang')
parser.add_argument('--seed', type=int, default=101, help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)

def mask_upper_left_corner(images, square_size = args.square_size):
    # Check if the square size is valid for the given images
    if square_size > images.shape[3] or square_size > images.shape[4]:
        raise ValueError("Square size is too large for the given images.")

    # Mask the upper-left corner
    images[:, :, :, :square_size, :square_size] = 0
    return images

def zoomin(images, crop_size=60, output_size=(84, 84)):
    """
    Efficiently crops the central part of each image in a sequence and resizes them to a given size.

    :param images: Tensor of shape (batch_size, sequence_length, channels, height, width)
    :param crop_size: Size of the square crop (height, width)
    :param output_size: Size of the output image after resizing (height, width)
    :return: Tensor of resized images
    """
    batch_size, sequence_length, channels, height, width = images.shape

    # Calculate the top-left pixel of the central crop
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2

    # Reshape and crop
    images = images.view(-1, channels, height, width)  # Combine batch and sequence dimensions
    cropped = F.crop(images, top, left, crop_size, crop_size)

    # Resize
    resized = F.resize(cropped, output_size)

    # Reshape back to original dimensions
    return resized.view(batch_size, sequence_length, channels, output_size[0], output_size[1])

def add_gaussian_noise(images, device, mean=0.0, std=0.03):
    """
    Adds Gaussian noise to a batch of images.

    :param images: Tensor of shape (batch_size, channels, height, width)
    :param mean: Mean of the Gaussian noise
    :param std: Standard deviation of the Gaussian noise
    :return: Tensor of images with added Gaussian noise
    """
    noise = torch.randn(images.size()) * std + mean
    noise = noise.to(device)
    noisy_images = images + noise
    # Clip the values to be within the valid range for images
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images

gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(2.0, 2.0))

class FisheyeEffect:
    def __init__(self, height, width, distortion_scale=0.5):
        # Create a grid representing the coordinate values of the original image
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
        # Convert to polar coordinates
        r = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx)
        # Fisheye mapping function using a polynomial transformation
        r_new = r + distortion_scale * r ** 3
        # Mask to limit the fisheye effect to a circle within the image
        mask = r <= 1.0
        # Convert back to cartesian coordinates
        xx_new = mask * r_new * torch.cos(theta)
        yy_new = mask * r_new * torch.sin(theta)
        # For points outside the circle, map them to the nearest border point
        xx_new[~mask] = torch.sign(xx[~mask])
        yy_new[~mask] = torch.sign(yy[~mask])
        # Scale back to image coordinates
        self.grid = torch.stack((xx_new, yy_new), dim=-1)

    def apply_fisheye_effect(self, images):
        B, C, H, W = images.shape
        device = images.device
        # Expand the grid to match the batch size of the images
        grid = self.grid.repeat(B, 1, 1, 1).to(device)
        # Apply grid sample using the expanded grid
        output_images = torch.nn.functional.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return output_images

fisheye_effect = FisheyeEffect(height=84, width=84)


if args.effect == 'mask':
    def add_effect(images):
        effect_images = mask_upper_left_corner(images)
        return effect_images
elif args.effect == 'zoomin':
    def add_effect(images):
        effect_images = zoomin(images)
        return effect_images
elif args.effect == 'noise':
    def add_effect(images):
        effect_images = add_gaussian_noise(images, device=device)
        return effect_images
elif args.effect == 'blur':
    def add_effect(images):
        batch_size, sequence_length, channels, height, width = images.shape
        images = images.view(-1, channels, height, width)  # Combine batch and sequence dimensions
        effect_images = gaussian_blur(images)
        return effect_images.view(batch_size, sequence_length, channels, height, width)
elif args.effect == 'fisheye':
    def add_effect(images):
        batch_size, sequence_length, channels, height, width = images.shape
        images = images.view(-1, channels, height, width)  # Combine batch and sequence dimensions
        effect_images = fisheye_effect.apply_fisheye_effect(images)
        return effect_images.view(batch_size, sequence_length, channels, height, width)
else:
    print("effect wrong.")

class ImitationLearningDataset(Dataset):
    def __init__(self, file_path, vision1, vision2, horizon=10, mask_name=None):
        super(ImitationLearningDataset, self).__init__()
        self.file = h5py.File(file_path, 'r')
        self.demos = [key for key in self.file['data'].keys() if "demo" in key]
        self.horizon = horizon
        self.vision1 = vision1
        self.vision2 = vision2

        # Apply mask if provided
        if mask_name:
            mask = self.file['mask'][mask_name][:]
            self.demos = [self.demos[i] for i in range(len(self.demos)) if i < len(mask) and mask[i]]

        self.data_points = []
        for demo_name in self.demos:
            demo = self.file['data'][demo_name]
            num_steps = demo['actions'].shape[0] - self.horizon + 1
            for step in range(num_steps):
                self.data_points.append((demo_name, step))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        demo_name, step = self.data_points[idx]
        demo = self.file['data'][demo_name]

        # Collect sequences of images, low_dim_obs, and actions
        seq_vision1_images = []
        seq_vision2_images = []
        seq_actions = []
        seq_eef_pos = []
        seq_eef_quat = []
        seq_gripper_qpos = []
        for i in range(self.horizon):
            current_step = step + i
            action = torch.tensor(demo['actions'][current_step], dtype=torch.float32)
            seq_actions.append(action)

            # Assuming the image observation is named 'agentview_image' and has shape (H, W, C)
            vision1_image = torch.tensor(demo['obs'][self.vision1 + '_image'][current_step], dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize and reshape to (C, H, W)
            vision2_image = torch.tensor(demo['obs'][self.vision2 + '_image'][current_step], dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize and reshape to (C, H, W)
            seq_vision1_images.append(vision1_image)
            seq_vision2_images.append(vision2_image)

            # Extract other low dimensional observations as needed
            # Example: eef_pos = torch.tensor(demo['obs']['robot0_eef_pos'][current_step], dtype=torch.float32)

            eef_pos = torch.tensor(demo['obs']['robot0_eef_pos'][current_step], dtype=torch.float32)
            eef_quat = torch.tensor(demo['obs']['robot0_eef_quat'][current_step], dtype=torch.float32)
            gripper_qpos = torch.tensor(demo['obs']['robot0_gripper_qpos'][current_step], dtype=torch.float32)

            seq_eef_pos.append(eef_pos)
            seq_eef_quat.append(eef_quat)
            seq_gripper_qpos.append(gripper_qpos)

        # Stack the sequences
        # size (horizon T: 10, channel C: 3, height H: 84 or 240, width W: 84 or 240)
        seq_vision1_images = torch.stack(seq_vision1_images)
        seq_vision2_images = torch.stack(seq_vision2_images)
        # size (horizon T: 10, length: 3)
        seq_eef_pos = torch.stack(seq_eef_pos)
        # size (horizon T: 10, length: 4)
        seq_eef_quat = torch.stack(seq_eef_quat)
        # size (horizon T: 10, length: 2)
        seq_gripper_qpos = torch.stack(seq_gripper_qpos)
        # size (horizon T: 10, length: 7)
        seq_actions = torch.stack(seq_actions)

        return seq_vision1_images, seq_vision2_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, seq_actions


# Full dataset
dataset_name = 'datasets/' + args.task + '/ph/' + args.dataset_name + '.hdf5'

dataset = ImitationLearningDataset(dataset_name, vision1=args.vision1, vision2=args.vision2)

dataset_train = ImitationLearningDataset(dataset_name, vision1=args.vision1, vision2=args.vision2, mask_name='train')
dataset_valid = ImitationLearningDataset(dataset_name, vision1=args.vision1, vision2=args.vision2, mask_name='valid')

print(len(dataset))
print(len(dataset_train))
print(len(dataset_valid))

data_loader_train = DataLoader(dataset=dataset_train, sampler=None, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
data_loader_valid = DataLoader(dataset=dataset_valid, sampler=None, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

# create environment from dataset
if args.task == 'tool_hang':
    camera_height = 240
    camera_width = 240
elif args.task == 'can' or args.task == 'square':
    camera_height = 84
    camera_width = 84
else:
    camera_height = -1
    camera_width = -1
    print("wrong task.")

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_name)
env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[args.vision1, args.vision2],
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
        use_depth_obs=False,
    )

# load the policy network
if args.task == 'tool_hang':
    input_shape = [512, 8, 8]
elif args.task == 'can' or args.task == 'square':
    input_shape = [512, 3, 3]
else:
    input_shape = None
    print("wrong task.")

image_latent_dim = args.anchor_num
action_dim = 7
low_dim_input_dim = 3 + 4 + 2  # robot0_eef_pos + robot0_eef_quat + robot0_gripper_qpos
rnn_hidden_dim = 1000

device = torch.device(args.device)

policy = PiNetwork(input_shape, image_latent_dim, action_dim, low_dim_input_dim, rnn_hidden_dim)
policy.to(device)
policy.float()

# start the training process

# Initialize the optimizer and validation loss criterion
optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=0.0)
eval_criterion = nn.MSELoss()

game_max_steps = args.horizon
games_num = 20
num_epochs = args.num_epochs
VALIDATION_INTERVAL = 10
TEST_ROLLOUT_INTERVAL = 10  # 10
rollout_successes = 0

if args.log:
    if args.process1 and not args.process2:
        writer = SummaryWriter('training_data/' + args.task + '/bc_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_view1' + args.effect)
    elif args.process2 and not args.process1:
        writer = SummaryWriter('training_data/' + args.task + '/bc_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_view2' + args.effect)
    elif args.process1 and args.process2:
        writer = SummaryWriter('training_data/' + args.task + '/bc_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_view12' + args.effect)
    else:
        writer = SummaryWriter('training_data/' + args.task + '/bc_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_no' + args.effect)

if args.save_model:
    model_path = 'saved_models/' + args.task + '/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if args.process1 and not args.process2:
        model_file_name = 'bc_rnn_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_view1' + args.effect + '_model.pt'
    elif args.process2 and not args.process1:
        model_file_name = 'bc_rnn_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_view2' + args.effect + '_model.pt'
    elif args.process1 and args.process2:
        model_file_name = 'bc_rnn_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_view12' + args.effect + '_model.pt'
    else:
        model_file_name = 'bc_rnn_robomimic_ver1_vision1' + args.vision1 + '_vision2' + args.vision2 + '_anchors' + str(args.anchor_num) + '_lr' + str(args.lr) + '_seed' + str(args.seed) + '_no' + args.effect + '_model.pt'



for epoch in range(num_epochs):
    # Training loop
    policy.train()
    running_loss = 0.0  # To accumulate the loss over batches
    num_batches = 0

    for data in data_loader_train:
        seq_vision1_images, seq_vision2_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, seq_actions = [d.to(device) for d in data]

        if args.process1:
            seq_vision1_images = add_effect(seq_vision1_images)
        if args.process2:
            seq_vision2_images = add_effect(seq_vision2_images)

        # Forward pass, default rnn_init_state=None, return_state=False
        action_dist = policy.forward_train(seq_vision1_images, seq_vision2_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos)
        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(action_dist.batch_shape) == 2  # [B, T]
        log_probs = action_dist.log_prob(seq_actions)
        # loss is just negative log-likelihood of action targets
        loss = -log_probs.mean()

        # backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_training_loss = running_loss / num_batches
    current_time = datetime.now()
    print(current_time.strftime("%m-%d %H:%M:%S") + f" - Training epoch {epoch} - Average Training Loss: {avg_training_loss:.4f}")

    # Add to tensorboard - Training
    if args.log:
        writer.add_scalar('average_training_loss', avg_training_loss, epoch)

    # Validation loop
    if (epoch+1) % VALIDATION_INTERVAL == 0:
        policy.eval()
        validation_loss = 0.0
        validation_num_batches = 0
        with torch.no_grad():
            for data in data_loader_valid:
                seq_vision1_images, seq_vision2_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, seq_actions = [d.to(device) for d in data]

                if args.process1:
                    seq_vision1_images = add_effect(seq_vision1_images)
                if args.process2:
                    seq_vision2_images = add_effect(seq_vision2_images)

                action_dist = policy.forward_train(seq_vision1_images, seq_vision2_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos)
                # make sure that this is a batch of multivariate action distributions, so that
                # the log probability computation will be correct
                assert len(action_dist.batch_shape) == 2  # [B, T]
                log_probs = action_dist.log_prob(seq_actions)
                # loss is just negative log-likelihood of action targets
                loss = -log_probs.mean()
                validation_loss += loss.item()
                validation_num_batches += 1
        avg_validation_loss = validation_loss / validation_num_batches
        print(f"Epoch {epoch}, Validation Loss: {avg_validation_loss}")

        # Add to tensorboard - Validation
        if args.log:
            writer.add_scalar('validation_loss', avg_validation_loss, epoch)

    # Testing loop (rollout), and save policy network parameters
    if (epoch + 1) % TEST_ROLLOUT_INTERVAL == 0:
        rollout_successes = 0
        policy.eval()
        with torch.no_grad():
            for game_i in range(games_num):
                obs = env.reset()
                rnn_state = None
                for step_i in range(game_max_steps):
                    # add two dimensions (batch size = 1, sequence length = 1) by two unsqueeze(0)
                    eef_pos = torch.tensor(obs['robot0_eef_pos'].copy(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    eef_quat = torch.tensor(obs['robot0_eef_quat'].copy(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    gripper_qpos = torch.tensor(obs['robot0_gripper_qpos'].copy(), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    vision1_images = torch.tensor(obs[args.vision1 + '_image'].copy(), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0
                    vision2_images = torch.tensor(obs[args.vision2 + '_image'].copy(), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0
                    if args.process1:
                        vision1_images = add_effect(vision1_images)
                    if args.process2:
                        vision2_images = add_effect(vision2_images)
                    # Predict action
                    # at the first step, rnn_state=None,
                    # and the policy will call self.get_rnn_init_state to get zeros rnn_state.
                    pi, rnn_state = policy.forward_step(vision1_images, vision2_images, eef_pos, eef_quat, gripper_qpos, rnn_init_state=rnn_state)
                    act = pi.cpu().squeeze().numpy()

                    # Environment step using the predicted action
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





