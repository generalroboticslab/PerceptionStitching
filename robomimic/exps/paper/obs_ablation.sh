#!/bin/bash

# ==========obs_ablation==========

#  task: square
#    dataset type: ph
#      hdf5 type: low_dim
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/low_dim/bc_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/low_dim/bc_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/low_dim/bc_rnn_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/low_dim/bc_rnn_add_proprio.json

#  task: square
#    dataset type: ph
#      hdf5 type: image
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/image/bc_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/image/bc_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/image/bc_remove_wrist.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/image/bc_remove_rand.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/image/bc_rnn_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/image/bc_rnn_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/image/bc_rnn_remove_wrist.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/ph/image/bc_rnn_remove_rand.json

#  task: square
#    dataset type: mh
#      hdf5 type: low_dim
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/low_dim/bc_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/low_dim/bc_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/low_dim/bc_rnn_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/low_dim/bc_rnn_add_proprio.json

#  task: square
#    dataset type: mh
#      hdf5 type: image
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/image/bc_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/image/bc_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/image/bc_remove_wrist.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/image/bc_remove_rand.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/image/bc_rnn_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/image/bc_rnn_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/image/bc_rnn_remove_wrist.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/square/mh/image/bc_rnn_remove_rand.json

#  task: transport
#    dataset type: ph
#      hdf5 type: low_dim
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/low_dim/bc_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/low_dim/bc_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/low_dim/bc_rnn_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/low_dim/bc_rnn_add_proprio.json

#  task: transport
#    dataset type: ph
#      hdf5 type: image
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/image/bc_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/image/bc_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/image/bc_remove_wrist.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/image/bc_remove_rand.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/image/bc_rnn_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/image/bc_rnn_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/image/bc_rnn_remove_wrist.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/ph/image/bc_rnn_remove_rand.json

#  task: transport
#    dataset type: mh
#      hdf5 type: low_dim
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/low_dim/bc_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/low_dim/bc_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/low_dim/bc_rnn_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/low_dim/bc_rnn_add_proprio.json

#  task: transport
#    dataset type: mh
#      hdf5 type: image
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/image/bc_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/image/bc_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/image/bc_remove_wrist.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/image/bc_remove_rand.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/image/bc_rnn_add_eef_vel.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/image/bc_rnn_add_proprio.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/image/bc_rnn_remove_wrist.json
python /home/pingcheng/PycharmProjects/robomimic/robomimic/scripts/train.py --config robomimic/exps/paper/obs_ablation/transport/mh/image/bc_rnn_remove_rand.json

