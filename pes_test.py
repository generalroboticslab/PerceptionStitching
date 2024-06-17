import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
# Go up one level to the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import robosuite
from robosuite.controllers import load_controller_config
from nn_modules2.resnet18_LSTMgmm_view13rgb_rel_model_ver1_out_xtask import PiNetwork
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import argparse
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

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
        default='cuda:0',
        help="the device for training",
    )

parser.add_argument(
        "--vision1",
        type=str,
        default='robot0_eye_in_hand',
        help="The image for encoder 1 of gmm mlp policy. Can be frontview, agentview, sideview, robot0_eye_in_hand.",
    )

parser.add_argument(
        "--vision2",
        type=str,
        default='agentview',
        help="The image for encoder 2 of gmm mlp policy. Can be frontview, agentview, sideview, robot0_eye_in_hand.",
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
        "--effect",
        type=str,
        default="mask",
        help="mask zoomin blur noise fisheye, mask default size is 14",
    )

parser.add_argument('--test_mode', type=int, default=1,
                    help='1 is for three parts from three different network.\
                         2 is for v1 and mlp from one network, v2 from another network.\
                         3 is for v2 and mlp from on network, v1 from another network.')

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

parser.add_argument('--anchor_num', type=int, default=512, help='number of anchors')
parser.add_argument('--games_num', type=int, default=50, help='number of games for testing')
parser.add_argument('--square_size', type=int, default=14, help='size of the square mask')
parser.add_argument('--seed', type=int, default=101, help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)

class CropRandomizer():
    def __init__(
        self,
        input_shape,
        crop_height=76,
        crop_width=76,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super(CropRandomizer, self).__init__()

        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width

    def random_crop(self, images):

        # Define the transformation - RandomCrop
        # Here the size of the crop is 76x76
        transform = transforms.RandomCrop([self.crop_height, self.crop_width])

        # Crop the images
        # Apply the transformation to each image in the batch
        cropped_images = torch.stack([transform(image) for image in images])

        return cropped_images

    def center_crop(self, images):
        """
        Center crops a batch of images to a specified size.

        :return: Tensor of cropped images
        """

        # Calculate the top left corner of the cropping area
        top = (self.input_shape[1] - self.crop_height) // 2  # self.input_shape[1] is input image height
        left = (self.input_shape[2] - self.crop_width) // 2  # self.input_shape[2] is input image width

        # Crop the images
        # since the input is batch of sequence of images,
        # with shape (batch_size, sequence_length, channel, height, width),
        # there should be three :, in the front, but not two of them in the BC training
        return images[:, :, :, top:top + self.crop_height, left:left + self.crop_width]

    def center_crop_anchor(self, images):
        """
        Center crops a batch of images to a specified size.

        :return: Tensor of cropped images
        """

        # Calculate the top left corner of the cropping area
        top = (self.input_shape[1] - self.crop_height) // 2  # self.input_shape[1] is input image height
        left = (self.input_shape[2] - self.crop_width) // 2  # self.input_shape[2] is input image width

        # Crop the images
        return images[:, :, top:top + self.crop_height, left:left + self.crop_width]


# create environment
dataset_name = 'datasets/' + args.task + '/ph/' + args.dataset_name + '.hdf5'
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

# load anchors
device = torch.device(args.device)
if args.vision1 == 'agentview':
    vision1_anchors = np.load('hdf5_image/can/' + args.vision1 + '_' +str(args.anchor_num) + 'anchor_images.npy')
else:
    vision1_anchors = np.load('hdf5_image/can/' + args.vision1 + '_' +str(args.anchor_num) + 'anchor_images_from_agentview_idx.npy')

if args.vision2 == 'agentview':
    vision2_anchors = np.load('hdf5_image/can/' + args.vision2 + '_' +str(args.anchor_num) + 'anchor_images.npy')
else:
    vision2_anchors = np.load('hdf5_image/can/' + args.vision2 + '_' +str(args.anchor_num) + 'anchor_images_from_agentview_idx.npy')

vision1_anchors_tensor = torch.tensor(vision1_anchors, dtype=torch.float32).to(device).permute(0, 3, 1, 2) / 255.0
vision2_anchors_tensor = torch.tensor(vision2_anchors, dtype=torch.float32).to(device).permute(0, 3, 1, 2) / 255.0

image_input_shape = [3, 84, 84]
crop_randomizer = CropRandomizer(input_shape=image_input_shape)

vision1_anchors_tensor = crop_randomizer.center_crop_anchor(vision1_anchors_tensor)
vision2_anchors_tensor = crop_randomizer.center_crop_anchor(vision2_anchors_tensor)

def mask_upper_left_corner(images, square_size = args.square_size):
    # Check if the square size is valid for the given images
    if square_size > images.shape[3] or square_size > images.shape[4]:
        raise ValueError("Square size is too large for the given images.")

    # Mask the upper-left corner
    images[:, :, :, :square_size, :square_size] = 0
    return images

def zoomin(images, crop_size=60, output_size=(76, 76)):
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

# gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(2.0, 2.0))
gaussian_blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0))

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

fisheye_effect = FisheyeEffect(height=76, width=76)


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


if args.process1:
    vision1_anchors_tensor = vision1_anchors_tensor.unsqueeze(0)
    vision1_anchors_tensor = add_effect(vision1_anchors_tensor)
    vision1_anchors_tensor = vision1_anchors_tensor.squeeze(0)
    print("process anchor vision1")
if args.process2:
    vision2_anchors_tensor = vision2_anchors_tensor.unsqueeze(0)
    vision2_anchors_tensor = add_effect(vision2_anchors_tensor)
    vision2_anchors_tensor = vision2_anchors_tensor.squeeze(0)
    print("process anchor vision2")

policy = PiNetwork(input_shape, vision1_anchors_tensor, vision2_anchors_tensor, image_latent_dim, action_dim, low_dim_input_dim, rnn_hidden_dim)
policy.to(device)
policy.float()

game_max_steps = 400
games_num = args.games_num

# test load and rollout

vision1_encoder_path = 'your_model_path'
vision2_encoder_path = 'your_model_path'
gmm_mlp_path = 'your_model_path'

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

repeat_num = 3
success_rate_list = []

with torch.no_grad():
    for i in range(repeat_num):
        rollout_successes = 0
        for game_i in range(games_num):
            print(game_i)
            obs = env.reset()
            rnn_state = None
            for step_i in range(game_max_steps):
                # add two dimensions (batch size = 1, sequence length = 1) by two unsqueeze(0)
                eef_pos = torch.tensor(obs['robot0_eef_pos'].copy(), dtype=torch.float32, device=device).unsqueeze(
                    0).unsqueeze(0)
                eef_quat = torch.tensor(obs['robot0_eef_quat'].copy(), dtype=torch.float32, device=device).unsqueeze(
                    0).unsqueeze(0)
                gripper_qpos = torch.tensor(obs['robot0_gripper_qpos'].copy(), dtype=torch.float32,
                                            device=device).unsqueeze(0).unsqueeze(0)
                vision1_images = torch.tensor(obs[args.vision1 + '_image'].copy(), dtype=torch.float32,
                                              device=device).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0
                vision2_images = torch.tensor(obs[args.vision2 + '_image'].copy(), dtype=torch.float32,
                                              device=device).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0

                vision1_images = crop_randomizer.center_crop(vision1_images)
                vision2_images = crop_randomizer.center_crop(vision2_images)
                if args.process1:
                    vision1_images = add_effect(vision1_images)
                if args.process2:
                    vision2_images = add_effect(vision2_images)

                # Predict action
                # at the first step, rnn_state=None,
                # and the policy will call self.get_rnn_init_state to get zeros rnn_state.
                pi, rnn_state = policy.forward_step(vision1_images, vision2_images, eef_pos, eef_quat, gripper_qpos,
                                                    rnn_init_state=rnn_state)
                act = pi.cpu().squeeze().numpy()

                # Environment step using the predicted action
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
