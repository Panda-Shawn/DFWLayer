# Differentiable Frank-Wolfe Optimization Layer

This repo contains code accompaning the paper, [Differentiable Frank-Wolfe Optimization Layer](https://arxiv.org/abs/2308.10806). DFWLayer is a differentiable optimization layer
which accelerates both the optimization and backprogation procedure with non-differentiable norm constraints.

## Dependencies

```
pip install -r requirements.txt
```

## Usage

### Different-Scale Optimization Problems

We test the efficiency (running time) and accuracy (simularity and distance) for different-scale optimization problems.
```
cd DFWLayer/numerical_experiment
python test_time_for_norms.py
```
The problem size can be changed by modifying `n=100` in `test_time_for_norms.py`.

### Robotics Tasks Under Imitation Learning

We evaluate the performance of differentiable optimization layers for robotics tasks under imitation learning.
1. The expert demonstrations are saved in `DFWLayer/robotics/expert_data`. We provide expert demonstartions for R+O03 and R+O10.
2. For example, we train policy for R+O03 with DFWLayer.
   ```
   cd DFWLayer/robotics
   python train_policy.py --cost_type R+O03 --opt_layer_class dfw_layer --device cuda
   ```
   The task and layer class can be changed by modifying arguments `--cost_type` and `--opt_layer_class` respectively.

## Citation
```
@article{liu2023differentiable,
  title={Differentiable Frank-Wolfe Optimization Layer},
  author={Liu, Zixuan and Liu, Liu and Wang, Xueqian and Zhao, Peilin},
  journal={arXiv preprint arXiv:2308.10806},
  year={2023}
}
```
