import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

import torchvision.models as models
from torchvision.models.convnext import ConvNeXt_Base_Weights, convnext_base, CNBlockConfig
from torchvision.models import resnet18
from torchvision.ops import Conv2dNormActivation
import torch.distributions as D


# In this version, I add relative representation to the original resnet18_gmmmlp_view13rgb_model_ver1.py
# Use four distance: Cos, L1, L2, L_infinite
# Aggregate the four distances with nonlinear layer + sum

class TanhWrappedDistribution(D.Distribution):
    """
    The TanhWrappedDistribution class is directly grabbed from the robomimic.models.distributions.
    https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/distributions.py
    Class that wraps another valid torch distribution, such that sampled values from the base distribution are
    passed through a tanh layer. The corresponding (log) probabilities are also modified accordingly.
    Tanh Normal distribution - adapted from rlkit and CQL codebase
    (https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/distributions.py#L6).
    """
    def __init__(self, base_dist, scale=1.0, epsilon=1e-6):
        """
        Args:
            base_dist (Distribution): Distribution to wrap with tanh output
            scale (float): Scale of output
            epsilon (float): Numerical stability epsilon when computing log-prob.
        """
        self.base_dist = base_dist
        self.scale = scale
        self.tanh_epsilon = epsilon
        super(TanhWrappedDistribution, self).__init__()

    def log_prob(self, value, pre_tanh_value=None):
        """
        Args:
            value (torch.Tensor): some tensor to compute log probabilities for
            pre_tanh_value: If specified, will not calculate atanh manually from @value. More numerically stable
        """
        value = value / self.scale
        if pre_tanh_value is None:
            one_plus_x = (1. + value).clamp(min=self.tanh_epsilon)
            one_minus_x = (1. - value).clamp(min=self.tanh_epsilon)
            pre_tanh_value = 0.5 * torch.log(one_plus_x / one_minus_x)
        lp = self.base_dist.log_prob(pre_tanh_value)
        tanh_lp = torch.log(1 - value * value + self.tanh_epsilon)
        # In case the base dist already sums up the log probs, make sure we do the same
        return lp - tanh_lp if len(lp.shape) == len(tanh_lp.shape) else lp - tanh_lp.sum(-1)

    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.base_dist.sample(sample_shape=sample_shape).detach()

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Sampling in the reparameterization case - for differentiable samples.
        """
        z = self.base_dist.rsample(sample_shape=sample_shape)

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev


class Resnet18SpatialSoftmax(nn.Module):
    """
    ResNet-18 network followed by a spatial-softmax layer, outputting a 64D vector.
    I don't add noise_std currently. noise_std is used to prevent over-fitting but don't use it yet,
    because noise_std=0.0 is in the default setting of robomimic, which is equivalent to adding no noise.
    I don't add output_variance currently, because the robomimic default value of output_variance is False,
    which means they also don't calculate and use this variance.
    """

    def __init__(self, input_shape, anchors_tensor, image_latent_dim=256, num_kp=32, temperature=1., learnable_temperature=False):
        super(Resnet18SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        self._num_kp = num_kp
        self.image_latent_dim = image_latent_dim
        self.anchors = anchors_tensor

        # Initialize ResNet-18 model without pretrained weights. weights=None means no pretrained parameters loaded.
        self.resnet18_base_model = resnet18(weights=None)  # pretrained=False is equivalent to weights=None
        self.features = nn.Sequential(*list(self.resnet18_base_model.children())[:-2])

        # Spatial Softmax
        self.spatial_softmax = nn.Conv2d(self._in_c, num_kp, kernel_size=1)

        # Temperature
        self.learnable_temperature = learnable_temperature
        if self.learnable_temperature:
            temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        # Keypoint position grid
        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., self._in_w), np.linspace(-1., 1., self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        # flatten the output keypoints from [B, K, 2] to [B, K * 2]
        self.flatten_kps = nn.Flatten(start_dim=1, end_dim=-1)

        # Fully connected layer to embed the flattened keypoints
        self.fc = nn.Linear(num_kp * 2, image_latent_dim)

        # the nonlinear layer for the aggregation of different distances
        self.num_subspaces = 4  # I use four different distances, L1, L2, Linf, Cos
        self.nonlinear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(image_latent_dim),
                    nn.Linear(image_latent_dim, image_latent_dim),
                    nn.Tanh(),
                )
                for _ in range(self.num_subspaces)
            ]
        )

    def forward(self, x):
        # Extract features using ResNet-18 base model
        x = torch.cat((x, self.anchors), 0)
        feature = self.features(x)
        assert (feature.shape[1] == self._in_c)
        assert (feature.shape[2] == self._in_h)
        assert (feature.shape[3] == self._in_w)

        # Apply spatial softmax
        feature = self.spatial_softmax(feature)

        # Reshape and softmax normalization
        feature = feature.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(feature / self.temperature, dim=-1)

        # Compute keypoints
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)
        feature_keypoints = self.flatten_kps(feature_keypoints)
        # Use normalize operation for the last layer of visual encoder
        feature_keypoints = F.normalize(self.fc(feature_keypoints), p=2.0, dim=1)
        # the x here is (16+64)*64 dim, because image_latent_dim=128
        # x_anchor is 256*128 dim, x_task_state is 128*128 dim
        x_task = feature_keypoints[0:-self.image_latent_dim, :]  # [0:-128, :]
        x_anchor = feature_keypoints[-self.image_latent_dim:, :]  # [-128:, :]
        # originally I need to do torch.nn.CosineSimilarity here
        # but notice that the x_anchor and x_task already have norm=1
        # I just need to do matrix multipilcation, and it will be equivalent to CosineSimilarity
        cos_distance = torch.mm(x_task, x_anchor.transpose(0, 1))
        # L1 distance
        l1_distance = torch.cdist(x_task, x_anchor, p=1)
        # L2 distance
        l2_distance = torch.cdist(x_task, x_anchor, p=2)
        # L infinite distance
        linf_distance = torch.cdist(x_task, x_anchor, p=float('inf'))

        # embed these distances through nonlinear layers
        cos_distance = self.nonlinear_layers[0](cos_distance)
        l1_distance = self.nonlinear_layers[1](l1_distance)
        l2_distance = self.nonlinear_layers[2](l2_distance)
        linf_distance = self.nonlinear_layers[3](linf_distance)

        distances = [cos_distance, l1_distance, l2_distance, linf_distance]

        return torch.stack(distances, dim=1).sum(dim=1)  # relative_interface


class GMM_MLP(nn.Module):
    def __init__(self, input_dim, low_dim_input_dim, output_dim=7, hidden_dims=[1024, 1024], num_modes=5, min_std=0.01, std_activation='softplus', low_noise_eval=True, use_tanh=False):
        super(GMM_MLP, self).__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh
        self.std_activation = std_activation

        self.low_dim_layer = nn.Linear(low_dim_input_dim, 64)

        # Define the MLP layers
        layers = []
        dims = [input_dim] + hidden_dims  # 3 for mean, scale, and logits
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(hidden_dims[-1], num_modes * output_dim)
        self.scale_layer = nn.Linear(hidden_dims[-1], num_modes * output_dim)
        self.logits_layer = nn.Linear(hidden_dims[-1], num_modes)

        # Activation function for scale
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

    def forward(self, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, emedded_robot0_eye_in_hand_image, emedded_agentview_image):
        low_dim_obs = torch.cat([robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos], 1)
        embedded_low_dim_obs = F.relu(self.low_dim_layer(low_dim_obs))

        x = torch.cat([embedded_low_dim_obs, emedded_robot0_eye_in_hand_image, emedded_agentview_image], 1)
        mlp_output = self.mlp(x)

        # Split output into mean, scale, and logits
        mean = self.mean_layer(mlp_output)
        scale = self.scale_layer(mlp_output)
        logits = self.logits_layer(mlp_output)

        # Process means and scales
        if not self.use_tanh:  # we use self.use_tanh = False as default, so will apply torch.tanh
            means = torch.tanh(mean.view(-1, self.num_modes, self.output_dim))  # Apply tanh normalization
        else:
            means = mean.view(-1, self.num_modes, self.output_dim)

        # Calculate scale
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            scales = self.activations[self.std_activation](scale.view(-1, self.num_modes, self.output_dim)) + self.min_std

        # Reshape logits, maybe don't need this line, since it is already the shape we want
        logits = logits.view(-1, self.num_modes)

        # Create the GMM
        component_distribution = D.Independent(D.Normal(loc=means, scale=scales), 1)
        mixture_distribution = D.Categorical(logits=logits)
        dist = D.MixtureSameFamily(mixture_distribution=mixture_distribution, component_distribution=component_distribution)

        # Optionally wrap with Tanh, default False
        if self.use_tanh:
            # Implement TanhWrappedDistribution here or use an existing implementation
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.)

        return dist


class PiNetwork(nn.Module):
    def __init__(self, input_shape, robot0_eye_in_hand_anchors_tensor, agentview_anchors_tensor, image_latent_dim, action_dim, low_dim_input_dim, mlp_hidden_dims):
        # env_params['action_max'] in gym is now env.action_spec[1], while [0] is action_min
        super(PiNetwork, self).__init__()

        self.RGBView1ResnetEmbed = Resnet18SpatialSoftmax(input_shape, robot0_eye_in_hand_anchors_tensor, image_latent_dim)
        self.RGBView3ResnetEmbed = Resnet18SpatialSoftmax(input_shape, agentview_anchors_tensor, image_latent_dim)
        self.Probot = GMM_MLP(input_dim=64 + 2 * image_latent_dim, low_dim_input_dim=low_dim_input_dim, output_dim=action_dim, hidden_dims=mlp_hidden_dims)

    def forward_train(self, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, robot0_eye_in_hand_image, agentview_image):
        """
                Return full GMM distribution, which is useful for computing
                quantities necessary at train-time, like log-likelihood, KL
                divergence, etc.

                Args:
                    low dimensional state of the proprioceptive observation and the image observations from view1 and view3

                Returns:
                    action_dist (Distribution): GMM distribution
                """
        emedded_robot0_eye_in_hand_image = self.RGBView1ResnetEmbed(robot0_eye_in_hand_image)
        emedded_agentview_image = self.RGBView3ResnetEmbed(agentview_image)
        action_dist = self.Probot(robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, emedded_robot0_eye_in_hand_image, emedded_agentview_image)
        return action_dist

    def forward(self, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, robot0_eye_in_hand_image, agentview_image):
        """
        Samples actions from the policy distribution.

        Args:
            low dimensional state of the proprioceptive observation and the image observations from view1 and view3

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        action_dist = self.forward_train(robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, robot0_eye_in_hand_image, agentview_image)
        return action_dist.sample()

