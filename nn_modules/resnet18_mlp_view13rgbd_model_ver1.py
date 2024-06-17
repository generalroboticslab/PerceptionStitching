import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import torchvision.models as models
from torchvision.models.convnext import ConvNeXt_Base_Weights, convnext_base, CNBlockConfig
from torchvision.models import resnet18
from torchvision.ops import Conv2dNormActivation


class Resnet18RGBEmbedding(nn.Module):
    def __init__(self, image_latent_dim=512):
        super(Resnet18RGBEmbedding, self).__init__()

        # Initialize ResNet-18 model without pretrained weights
        self.resnet18_base_model = resnet18(weights=None)

        # Replace the classifier head to match the desired output dimension
        self.resnet18_base_model.fc = nn.Linear(self.resnet18_base_model.fc.in_features, image_latent_dim)

    def forward(self, x):
        # Pass input through ResNet-18 base model
        x = self.resnet18_base_model(x)
        return x


class Resnet18DepthEmbedding(nn.Module):
    def __init__(self, depth_latent_dim=512):
        super(Resnet18DepthEmbedding, self).__init__()

        # Initialize ResNet-18 model without pretrained weights
        self.resnet18_base_model = resnet18(weights=None)

        # Modify the first convolutional layer for 1-channel input
        self.resnet18_base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the classifier head to match the desired output dimension
        self.resnet18_base_model.fc = nn.Linear(self.resnet18_base_model.fc.in_features, depth_latent_dim)

    def forward(self, x):
        # Pass input through ResNet-18 base model
        x = self.resnet18_base_model(x)
        return x


class MLP(nn.Module):
    # this is copied from MLP_ver5, which should be the same as MLP_ver3
    # output the mean and std for the gaussian distribution of the action, for SAC algorithm
    # The module for processing the robot info. Pi (policy) network can use this module.
    # interface_dim is the dimension of the interface between task and robot module. 16.
    # num_action is 3 for reach and push, 4 for pick
    # num_robot_inputs is 7 for reach and push, 8 for pick
    def __init__(self, num_actions, low_dim_input_dim, low_dim_embed_dim, fuse_view_embed_dim, image_latent_dim, depth_latent_dim, mlp_hidden_dim, env_action_max):
        super(MLP, self).__init__()

        self.max_action = env_action_max

        # Normalization layer for input low dim
        self.norm_input = nn.BatchNorm1d(low_dim_input_dim)
        self.linear_embed_low_dim = nn.Linear(low_dim_input_dim, low_dim_embed_dim)
        self.linear_fuse_view1 = nn.Linear(image_latent_dim + depth_latent_dim, fuse_view_embed_dim)
        self.linear_fuse_view3 = nn.Linear(image_latent_dim + depth_latent_dim, fuse_view_embed_dim)
        self.linear_fuse_high_dim = nn.Linear(2 * fuse_view_embed_dim, 2 * fuse_view_embed_dim)

        self.linear1 = nn.Linear(low_dim_embed_dim + 2 * fuse_view_embed_dim, mlp_hidden_dim)
        self.linear2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.linear3 = nn.Linear(mlp_hidden_dim, num_actions)

    def forward(self, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, emedded_robot0_eye_in_hand_image, emedded_agentview_image, emedded_robot0_eye_in_hand_depth, emedded_agentview_depth):
        low_dim = torch.cat([robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos], 1)
        low_dim = self.norm_input(low_dim)  # Normalize low_dim
        low_dim = F.relu(self.linear_embed_low_dim(low_dim))

        view1 = torch.cat([emedded_robot0_eye_in_hand_image, emedded_robot0_eye_in_hand_depth], 1)
        view3 = torch.cat([emedded_agentview_image, emedded_agentview_depth], 1)
        view1 = F.relu(self.linear_fuse_view1(view1))
        view3 = F.relu(self.linear_fuse_view3(view3))

        view13 = torch.cat([view1, view3], 1)
        view13 = F.relu(self.linear_fuse_high_dim(view13))

        x = torch.cat([low_dim, view13], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        action = self.max_action * torch.tanh(self.linear3(x))

        return action

class PiNetwork_ver1(nn.Module):
    def __init__(self, image_latent_dim, depth_latent_dim, num_actions, low_dim_input_dim, low_dim_embed_dim, fuse_view_embed_dim, mlp_hidden_dim, env_action_max):
        # env_params['action_max'] in gym is now env.action_spec[1], while [0] is action_min
        super(PiNetwork_ver1, self).__init__()

        self.RGBView1ResEmbed = Resnet18RGBEmbedding(image_latent_dim)
        self.RGBView3ResEmbed = Resnet18RGBEmbedding(image_latent_dim)
        self.DepthView1ResEmbed = Resnet18DepthEmbedding(depth_latent_dim)
        self.DepthView3ResEmbed = Resnet18DepthEmbedding(depth_latent_dim)
        self.Probot = MLP(num_actions, low_dim_input_dim, low_dim_embed_dim, fuse_view_embed_dim, image_latent_dim, depth_latent_dim, mlp_hidden_dim, env_action_max)

    def forward(self, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, robot0_eye_in_hand_image, agentview_image, robot0_eye_in_hand_depth, agentview_depth):
        emedded_robot0_eye_in_hand_image = self.RGBView1ResEmbed(robot0_eye_in_hand_image)
        emedded_agentview_image = self.RGBView3ResEmbed(agentview_image)
        emedded_robot0_eye_in_hand_depth = self.DepthView1ResEmbed(robot0_eye_in_hand_depth)
        emedded_agentview_depth = self.DepthView3ResEmbed(agentview_depth)
        action = self.Probot(robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, emedded_robot0_eye_in_hand_image, emedded_agentview_image, emedded_robot0_eye_in_hand_depth, emedded_agentview_depth)
        return action

