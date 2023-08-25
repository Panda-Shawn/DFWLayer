import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)
from stable_baselines3.common.torch_layers import FlattenExtractor
from util import *
import numpy as np
import pickle as pkl
from opt_actor import Cost, OptActor
from opt_layer_robotics import cvxpy_layer, altdiff_layer, dfw_layer
import argparse
import os 
import time
import random
import time


def make_actor(net_kwargs=None, features_extractor_class=None, features_extractor_kwargs={}, device="cpu"):
    net_kwargs = net_kwargs.copy()
    if features_extractor_class is None:
        features_extractor = FlattenExtractor(net_kwargs["observation_space"], **features_extractor_kwargs)
    net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
    return OptActor(**net_kwargs).to(device)

def make_train_dataset(replay_buffer, ratio):
    pos = int(replay_buffer.size() * ratio)
    replay_buffer.observations = replay_buffer.observations[:pos]
    replay_buffer.dones = replay_buffer.dones[:pos]
    replay_buffer.actions = replay_buffer.actions[:pos]
    replay_buffer.next_observations = replay_buffer.next_observations[:pos]
    replay_buffer.rewards = replay_buffer.rewards[:pos]
    replay_buffer.full = False
    replay_buffer.pos = pos

def make_test_dataset(replay_buffer, ratio):
    pos = int(replay_buffer.size() * ratio)
    replay_buffer.observations = replay_buffer.observations[pos:]
    replay_buffer.dones = replay_buffer.dones[pos:]
    replay_buffer.actions = replay_buffer.actions[pos:]
    replay_buffer.next_observations = replay_buffer.next_observations[pos:]
    replay_buffer.rewards = replay_buffer.rewards[pos:]
    replay_buffer.full = False
    replay_buffer.pos = replay_buffer.buffer_size - pos

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./expert_data/")
    parser.add_argument("--cost_type", default="HC+O", choices=["HC+O", "R+O10", "R+O03"])
    parser.add_argument("--opt_layer_class", default="cvxpy_layer", choices=["cvxpy_layer", "dfw_layer", "altdiff_layer"])
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


if __name__ == "__main__":
    args = add_parser()
    
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)

    file_path = os.path.join(args.data_path, f"{args.cost_type}-DPro-1.pkl")
    cost_type = args.cost_type
    cost = Cost(cost_type)
    with open(file_path, "rb") as f:
        train_buffer = pkl.load(f)
    make_train_dataset(train_buffer, ratio=args.ratio)

    with open(file_path, "rb") as f:
        test_buffer = pkl.load(f)
    make_test_dataset(test_buffer, ratio=args.ratio)
    test_data = test_buffer.sample(1000)

    if args.opt_layer_class=="cvxpy_layer":
        opt_layer_class = cvxpy_layer
    elif args.opt_layer_class=="altdiff_layer":
        opt_layer_class = altdiff_layer
    elif args.opt_layer_class=="dfw_layer":
        opt_layer_class = dfw_layer
    else:
        raise ValueError("Unknown optimization layer!")
    

    net_kwargs = {
        "observation_space": train_buffer.observation_space,
        "action_space": train_buffer.action_space,
        "net_arch": [400, 300],
        "activation_fn": torch.nn.ReLU,
        "normalize_images": True,
        "cost": cost,
        "opt_layer_class": opt_layer_class,
        "opt_layer_eps": args.eps,
    }


    policy = make_actor(
        net_kwargs=net_kwargs,
        features_extractor_class=None,
        features_extractor_kwargs={},
        device=device
    )
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    n_epochs = args.n_epochs
    n_steps = args.n_steps
    batch_size = args.batch_size
    save_dir = os.path.join(
        args.cost_type,
        args.opt_layer_class,
        time.strftime("%a-%b-%d-%H:%M:%S-%Y", time.localtime())
    )
    os.makedirs(save_dir, exist_ok=True)
    
    print("start training!")
    test_losses = []
    costs = []
    violation_rates = []
    times = []
    print("test before training!")
    epoch_test_loss = []
    epoch_test_cost = []
    epoch_test_vio = []
    with torch.no_grad():
        for k in range(5):
            obs = test_data.observations[200*k:200*(k+1), :]
            acts = test_data.actions[200*k:200*(k+1), :]
            pro_acts = policy(obs)
            loss = loss_fn(pro_acts, acts)
            epoch_test_loss.append(loss.item())
            cost, violation_rate, max_cost = policy.cost.compute_cost(pro_acts, obs, cost_type)
            epoch_test_cost.append(cost.item())
            epoch_test_vio.append(violation_rate.item())
        
    print(f"test loss:{loss.item()}, mean cost: {cost.item()}, violation rate: {violation_rate}")
    test_losses.append(epoch_test_loss)
    costs.append(epoch_test_cost)
    violation_rates.append(epoch_test_vio)
    
    for i in range(n_epochs):
        losses = []
        
        for j in range(n_steps):
            begin = time.time()
            batch_data = train_buffer.sample(batch_size)
            pro_acts = policy(batch_data.observations)
            loss = loss_fn(pro_acts, batch_data.actions)
            losses.append(loss.item())
            cost, violation_rate, max_cost = policy.cost.compute_cost(pro_acts, batch_data.observations, cost_type)
            loss += 0.00 * cost
            policy_optimizer.zero_grad()
            loss.backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip)
            policy_optimizer.step()
            stop = time.time()
            times.append(stop-begin)
            # print("training batch time:", stop-begin)

            # losses.append(loss.item())

        print(f"epoch: {i}, loss: {np.mean(losses)}")
        np.save(os.path.join(save_dir, f"train_loss_epoch_{i}.npy"), losses)
        if i % 20 == 19:
            print("start testing!")
            # test_data = test_buffer.sample(test_buffer.size())
            # test_data = test_buffer.sample(200)
            epoch_test_loss = []
            epoch_test_cost = []
            epoch_test_vio = []
            with torch.no_grad():
                for k in range(5):
                    obs = test_data.observations[200*k:200*(k+1), :]
                    acts = test_data.actions[200*k:200*(k+1), :]
                    pro_acts = policy(obs)
                    loss = loss_fn(pro_acts, acts)
                    epoch_test_loss.append(loss.item())
                    cost, violation_rate, max_cost = policy.cost.compute_cost(pro_acts, obs, cost_type)
                    epoch_test_cost.append(cost.item())
                    epoch_test_vio.append(violation_rate.item())
            test_losses.append(epoch_test_loss)
            costs.append(epoch_test_cost)
            violation_rates.append(epoch_test_vio)

            print(f"epoch: {i}, test loss:{loss.item()}, mean cost: {cost.item()}, violation rate: {violation_rate}, max cost: {max_cost.item()}")
    np.save(os.path.join(save_dir, f"test_loss.npy"), test_losses)
    np.save(os.path.join(save_dir, f"test_cost.npy"), costs)
    np.save(os.path.join(save_dir, f"test_violation_rate.npy"), violation_rates)
    np.save(os.path.join(save_dir, f"train_time.npy"), times)
    