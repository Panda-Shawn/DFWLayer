from typing import Any, Dict, List, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp


class Cost:
    def __init__(self, cost_type="HC+O") -> None:
        if cost_type=="HC+O":
            self.offset = 11
            self.a_dim = 6
            self.s_dim = 17
            self.max_power = 20.0
            self.indices = list(range(self.offset, self.offset+self.a_dim))
        elif cost_type=="R+O10":
            self.offset = 6
            self.a_dim = 2
            self.s_dim = 9
            self.max_power = 1.0
            self.indices = [6, 8]
        elif cost_type=="R+O03":
            self.offset = 6
            self.a_dim = 2
            self.s_dim = 9  
            self.max_power = 0.3
            self.indices = [6, 8]
        else:
            raise ValueError("Unknown cost type!")

    def compute_cost(self, acts=None, states=None, cost_type="HC+O"):
        if cost_type=="HC+O":
            violation = th.relu(th.sum(th.abs(states[:, self.indices] * acts), dim=1) - self.max_power - 1e-5)
            violation_count = th.sum(violation > 0)
            max_cost = th.max(violation)
            cost = th.sum(violation) / (violation_count+1e-8)
            violation_rate = violation_count / acts.shape[0]
        elif cost_type=="R+O10":
            violation = th.relu(th.sum(th.abs(states[:, self.indices] * acts), dim=1) - self.max_power)
            violation_count = th.sum(violation > 0)
            cost = th.sum(violation) / (violation_count+1e-8)
            violation_rate = violation_count / acts.shape[0]
        elif cost_type=="R+O03":
            violation = th.relu(th.sum(th.abs(states[:, self.indices] * acts), dim=1) - self.max_power)
            violation_count = th.sum(violation > 0)
            cost = th.sum(violation) / (violation_count+1e-8)
            violation_rate = violation_count / acts.shape[0]
        return cost, violation_rate, max_cost


class OptActor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        cost: Cost=None,
        opt_layer_class=None,
        opt_layer_eps=1e-3,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)
        self.cost = cost
        self.opt_layer = None
        if opt_layer_class is not None:
            self.opt_layer = opt_layer_class(eps=opt_layer_eps)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        if self.opt_layer is None:
            return self.mu(features)
        pre_opt = self.mu(features)
        tmp = th.ones(pre_opt.shape[1])
        P = th.diag(tmp)
        q = -pre_opt
        w = obs.float()[:,self.cost.indices].detach()
        t = th.tensor(self.cost.max_power)
        P = P.unsqueeze(0).expand(pre_opt.shape[0], pre_opt.shape[1], pre_opt.shape[1]).to(pre_opt.device)
        t = t.unsqueeze(0).expand(pre_opt.shape[0], 1).to(pre_opt.device)
        return self.opt_layer(P, q, w, t)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)
    
