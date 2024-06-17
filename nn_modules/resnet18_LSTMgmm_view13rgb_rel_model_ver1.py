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


# In this version, I try to strictly reproduce the network structure in the Robomimic for bc_rnn training
# the TanhWrappedDistribution class is directly grabbed from the robomimic.models.distributions
# https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/distributions.py
# default to have low dimension embedding layer
# add anchors and relative representation for the resnet
class TanhWrappedDistribution(D.Distribution):
    """
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

    def __init__(self, input_shape, anchors_tensor, image_latent_dim=512, num_kp=32, temperature=1., learnable_temperature=False):
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
        relative_interface = torch.mm(x_task, x_anchor.transpose(0, 1))

        return relative_interface


class LSTM_GMM(nn.Module):
    def __init__(self, input_dim, low_dim_input_dim, output_dim=7, rnn_hidden_dim=1000, rnn_num_layers=2, rnn_bidirectional=False, num_modes=5, min_std=0.01, std_activation='softplus', low_noise_eval=True, use_tanh=False):
        super(LSTM_GMM, self).__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh
        self.std_activation = std_activation
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.low_dim_layer = nn.Linear(low_dim_input_dim, 64)

        # LSTM Network
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=rnn_bidirectional
        )

        # Calculate output dimension considering bidirectional LSTM
        num_directions = 2 if rnn_bidirectional else 1
        lstm_output_dim = rnn_hidden_dim * num_directions

        self.rnn_num_directions = num_directions

        # GMM Layers
        self.mean_layer = nn.Linear(lstm_output_dim, num_modes * output_dim)
        self.scale_layer = nn.Linear(lstm_output_dim, num_modes * output_dim)
        self.logits_layer = nn.Linear(lstm_output_dim, num_modes)

        # Activation function for scale
        self.activations = {"softplus": F.softplus, "exp": torch.exp}

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)
        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        h_0 = torch.zeros(self.rnn_num_layers * self.rnn_num_directions, batch_size, self.rnn_hidden_dim).to(device)
        c_0 = torch.zeros(self.rnn_num_layers * self.rnn_num_directions, batch_size, self.rnn_hidden_dim).to(device)
        return h_0, c_0

    def rnn_forward(self, low_dim_obs, embedded_images, rnn_init_state=None, return_state=False):
        """
        Forward a sequence of inputs through the RNN and the per-step network.

        Args:
            inputs (torch.Tensor): tensor input of shape [B, T, D], where D is the RNN input size

            rnn_init_state: rnn hidden state, initialize to zero state if set to None

            return_state (bool): whether to return hidden state

        Returns:
            outputs: outputs of the per_step_net

            rnn_state: return rnn state at the end if return_state is set to True
        """
        embedded_low_dim_obs = F.relu(self.low_dim_layer(low_dim_obs))
        inputs = torch.cat([embedded_low_dim_obs, embedded_images], 2)

        assert inputs.ndimension() == 3  # [B, T, D]
        batch_size, seq_length, inp_dim = inputs.shape
        if rnn_init_state is None:
            rnn_init_state = self.get_rnn_init_state(batch_size, device=inputs.device)

        rnn_outputs, rnn_state = self.lstm(inputs, rnn_init_state)
        # reshape the rnn_outputs tensor of shape [B, T, lstm_output_dim] into shape [B*T, lstm_output_dim]
        rnn_outputs = rnn_outputs.reshape(batch_size * seq_length, -1)

        # Split output into mean, scale, and logits.
        mean = self.mean_layer(rnn_outputs)
        scale = self.scale_layer(rnn_outputs)
        logits = self.logits_layer(rnn_outputs)

        # Process means
        if not self.use_tanh:  # we use self.use_tanh = False as default, so will apply torch.tanh
            means = torch.tanh(mean.view(-1, self.num_modes, self.output_dim))  # Apply tanh normalization
        else:
            means = mean.view(-1, self.num_modes, self.output_dim)

        # Calculate scale
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            scales = self.activations[self.std_activation](
                scale.view(-1, self.num_modes, self.output_dim)) + self.min_std

        # Reshape logits, maybe don't need this line, since it is already the shape we want
        logits = logits.view(-1, self.num_modes)

        # Reshape the means tensor of shape [B*T, self.num_modes, self.output_dim]
        # into shape [B, T, self.num_modes, self.output_dim]
        means = means.view(batch_size, seq_length, self.num_modes, self.output_dim)

        # Reshape the scales tensor of shape [B*T, self.num_modes, self.output_dim]
        # into shape [B, T, self.num_modes, self.output_dim]
        scales = scales.view(batch_size, seq_length, self.num_modes, self.output_dim)

        # Reshape the logits tensor of shape [B*T, self.num_modes]
        # into shape [B, T, self.num_modes]
        logits = logits.view(batch_size, seq_length, self.num_modes)

        if return_state:
            return means, scales, logits, rnn_state
        else:
            return means, scales, logits

    def forward(self, low_dim_obs, embedded_images, rnn_init_state=None, return_state=False):
        # LSTM Forward Pass
        if return_state:
            means, scales, logits, rnn_state = self.rnn_forward(low_dim_obs, embedded_images, rnn_init_state=rnn_init_state, return_state=return_state)
        else:
            means, scales, logits= self.rnn_forward(low_dim_obs, embedded_images, rnn_init_state=rnn_init_state, return_state=return_state)

        # Creating the GMM
        component_distribution = D.Independent(D.Normal(loc=means, scale=scales), 1)
        mixture_distribution = D.Categorical(logits=logits)
        dist = D.MixtureSameFamily(mixture_distribution=mixture_distribution, component_distribution=component_distribution)

        # Optionally wrap with Tanh, default False
        if self.use_tanh:
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.)

        if return_state:
            return dist, rnn_state
        else:
            return dist


class PiNetwork(nn.Module):
    def __init__(self, input_shape, robot0_eye_in_hand_anchors_tensor, agentview_anchors_tensor, image_latent_dim, action_dim, low_dim_input_dim, rnn_hidden_dim):
        super(PiNetwork, self).__init__()

        self.image_latent_dim = image_latent_dim
        self.RGBView1ResnetEmbed = Resnet18SpatialSoftmax(input_shape, robot0_eye_in_hand_anchors_tensor, image_latent_dim)
        self.RGBView3ResnetEmbed = Resnet18SpatialSoftmax(input_shape, agentview_anchors_tensor, image_latent_dim)
        self.Probot = LSTM_GMM(input_dim=64 + 2 * image_latent_dim, low_dim_input_dim=low_dim_input_dim, output_dim=action_dim, rnn_hidden_dim=rnn_hidden_dim)

    def forward_train(self, seq_agentview_images, seq_robot0_eye_in_hand_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, rnn_init_state=None, return_state=False):
        """
            Return full GMM distribution, which is useful for computing
            quantities necessary at train-time, like log-likelihood, KL
            divergence, etc.

            Args:
                low dimensional state of the proprioceptive observation and the image observations from view1 and view3

            Returns:
                action_dist (Distribution): GMM distribution
        """
        # Embedding images using ResNet18SpatialSoftmax
        batch_size, seq_len, C, H, W = seq_agentview_images.shape
        seq_agentview_images = seq_agentview_images.view(-1, C, H, W)  # Combine batch and sequence dimensions
        seq_robot0_eye_in_hand_images = seq_robot0_eye_in_hand_images.view(-1, C, H, W)  # Combine batch and sequence dimensions
        embedded_view1 = self.RGBView1ResnetEmbed(seq_agentview_images)
        embedded_view3 = self.RGBView3ResnetEmbed(seq_robot0_eye_in_hand_images)
        embedded_view1 = embedded_view1.view(batch_size, seq_len, self.image_latent_dim)
        embedded_view3 = embedded_view3.view(batch_size, seq_len, self.image_latent_dim)

        embedded_images = torch.cat([embedded_view1, embedded_view3], dim=2)

        # Concatenating embedded images with low dimensional observations
        low_dim_obs = torch.cat([seq_eef_pos, seq_eef_quat, seq_gripper_qpos], dim=2)

        # LSTM_GMM Forward Pass
        if return_state:
            action_dist, rnn_state = self.Probot(low_dim_obs, embedded_images, rnn_init_state=rnn_init_state, return_state=return_state)
            return action_dist, rnn_state
        else:
            action_dist = self.Probot(low_dim_obs, embedded_images, rnn_init_state=rnn_init_state, return_state=return_state)
            return action_dist

    def forward(self, seq_agentview_images, seq_robot0_eye_in_hand_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, rnn_init_state=None, return_state=False):
        """
            Samples actions from the policy distribution.

            Args:
                low dimensional state of the proprioceptive observation and the image observations from view1 and view3

            Returns:
                action (torch.Tensor): batch of actions from policy distribution
        """
        if return_state:
            action_dist, rnn_state = self.forward_train(seq_agentview_images, seq_robot0_eye_in_hand_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, rnn_init_state=rnn_init_state, return_state=return_state)
            return action_dist.sample(), rnn_state
        else:
            action_dist = self.forward_train(seq_agentview_images, seq_robot0_eye_in_hand_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, rnn_init_state=rnn_init_state, return_state=return_state)
            return action_dist.sample()

    def forward_step(self, seq_agentview_images, seq_robot0_eye_in_hand_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, rnn_init_state=None):
        """
        Unroll RNN over single timestep to get sampled actions.
        """
        acts, rnn_state = self.forward(seq_agentview_images, seq_robot0_eye_in_hand_images, seq_eef_pos, seq_eef_quat, seq_gripper_qpos, rnn_init_state=rnn_init_state, return_state=True)
        assert acts.shape[1] == 1
        return acts[:, 0], rnn_state

