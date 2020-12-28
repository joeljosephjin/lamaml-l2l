# lamaml-l2l

Trying to implement La-MAML in Learn2Learn library framework.

'''
Set seed 0
/usr/local/lib/python3.6/dist-packages/torch/nn/modules/container.py:434: UserWarning: Setting attributes on ParameterList is not supported.
  warnings.warn("Setting attributes on ParameterList is not supported.")
Task: 0 | Epoch: 1/1 | Iter: 4 | Loss: 2.043 | Acc: Total: 0.10325 Current Task: 0.1072 : 100% 5/5 [00:01<00:00,  2.64it/s]
Task: 1 | Epoch: 1/1 | Iter: 4 | Loss: 1.464 | Acc: Total: 0.18083 Current Task: 0.3208 : 100% 5/5 [00:01<00:00,  2.63it/s]
Task: 2 | Epoch: 1/1 | Iter: 4 | Loss: 1.886 | Acc: Total: 0.22473 Current Task: 0.5368 : 100% 5/5 [00:01<00:00,  2.62it/s]
Task: 3 | Epoch: 1/1 | Iter: 4 | Loss: 1.294 | Acc: Total: 0.23533 Current Task: 0.4319 : 100% 5/5 [00:01<00:00,  2.56it/s]
Task: 4 | Epoch: 1/1 | Iter: 4 | Loss: 1.007 | Acc: Total: 0.28168 Current Task: 0.4289 : 100% 5/5 [00:01<00:00,  2.58it/s]
Task: 5 | Epoch: 1/1 | Iter: 4 | Loss: 1.02 | Acc: Total: 0.31133 Current Task: 0.4567 : 100% 5/5 [00:01<00:00,  2.58it/s]
Task: 6 | Epoch: 1/1 | Iter: 4 | Loss: 1.244 | Acc: Total: 0.34235 Current Task: 0.441 : 100% 5/5 [00:01<00:00,  2.60it/s]
Task: 7 | Epoch: 1/1 | Iter: 4 | Loss: 0.96 | Acc: Total: 0.35949 Current Task: 0.4456 : 100% 5/5 [00:01<00:00,  2.54it/s]
Task: 8 | Epoch: 1/1 | Iter: 4 | Loss: 1.025 | Acc: Total: 0.37889 Current Task: 0.3512 : 100% 5/5 [00:01<00:00,  2.58it/s]
Task: 9 | Epoch: 1/1 | Iter: 4 | Loss: 1.346 | Acc: Total: 0.40422 Current Task: 0.3211 : 100% 5/5 [00:01<00:00,  2.58it/s]
Task: 10 | Epoch: 1/1 | Iter: 4 | Loss: 1.0 | Acc: Total: 0.43811 Current Task: 0.3904 : 100% 5/5 [00:01<00:00,  2.59it/s]
Task: 11 | Epoch: 1/1 | Iter: 4 | Loss: 0.824 | Acc: Total: 0.46452 Current Task: 0.3172 : 100% 5/5 [00:01<00:00,  2.54it/s]
Task: 12 | Epoch: 1/1 | Iter: 4 | Loss: 0.625 | Acc: Total: 0.48317 Current Task: 0.3543 : 100% 5/5 [00:01<00:00,  2.58it/s]
Task: 13 | Epoch: 1/1 | Iter: 4 | Loss: 0.826 | Acc: Total: 0.5113 Current Task: 0.3383 : 100% 5/5 [00:01<00:00,  2.57it/s]
Task: 14 | Epoch: 1/1 | Iter: 4 | Loss: 0.687 | Acc: Total: 0.53477 Current Task: 0.3236 : 100% 5/5 [00:01<00:00,  2.57it/s]
Task: 15 | Epoch: 1/1 | Iter: 4 | Loss: 1.28 | Acc: Total: 0.55233 Current Task: 0.2568 : 100% 5/5 [00:01<00:00,  2.54it/s]
Task: 16 | Epoch: 1/1 | Iter: 4 | Loss: 0.798 | Acc: Total: 0.57987 Current Task: 0.298 : 100% 5/5 [00:01<00:00,  2.57it/s]
Task: 17 | Epoch: 1/1 | Iter: 4 | Loss: 1.025 | Acc: Total: 0.59714 Current Task: 0.3817 : 100% 5/5 [00:01<00:00,  2.57it/s]
Task: 18 | Epoch: 1/1 | Iter: 4 | Loss: 0.982 | Acc: Total: 0.63528 Current Task: 0.3947 : 100% 5/5 [00:01<00:00,  2.54it/s]
Task: 19 | Epoch: 1/1 | Iter: 4 | Loss: 1.141 | Acc: Total: 0.64562 Current Task: 0.3527 : 100% 5/5 [00:01<00:00,  2.54it/s]
####Final Validation Accuracy####
Final Results:- 
 Total Accuracy: 0.6604299545288086 
 Individual Accuracy: [tensor(0.6190), tensor(0.6631), tensor(0.6915), tensor(0.7214), tensor(0.7241), tensor(0.7297), tensor(0.7309), tensor(0.7206), tensor(0.7257), tensor(0.7163), tensor(0.7133), tensor(0.6859), tensor(0.6777), tensor(0.6678), tensor(0.6570), tensor(0.6263), tensor(0.6082), tensor(0.5797), tensor(0.5336), tensor(0.4168)]
logs//lamaml/test_lamaml-2020-12-28_13-51-14-6383/0/results: {'expt_name': 'test_lamaml', 'model': 'lamaml', 'arch': 'linear', 'n_hiddens': 100, 'n_layers': 2, 'xav_init': False, 'glances': 5, 'n_epochs': 1, 'batch_size': 10, 'replay_batch_size': 10.0, 'memories': 200, 'lr': 0.001, 'cuda': False, 'seed': 0, 'log_every': 100, 'log_dir': 'logs//lamaml/test_lamaml-2020-12-28_13-51-14-6383/0', 'tf_dir': 'logs//lamaml/test_lamaml-2020-12-28_13-51-14-6383/0/tfdir', 'calc_test_accuracy': False, 'data_path': 'data/', 'loader': 'task_incremental_loader', 'samples_per_task': 50, 'shuffle_tasks': False, 'classes_per_it': 4, 'iterations': 5000, 'dataset': 'mnist_rotations', 'workers': 3, 'validation': 0.0, 'class_order': 'old', 'increment': 5, 'test_batch_size': 100000, 'opt_lr': 0.3, 'opt_wt': 0.1, 'alpha_init': 0.15, 'learn_lr': True, 'sync_update': False, 'grad_clip_norm': 2.0, 'cifar_batches': 3, 'use_old_task_memory': True, 'second_order': False, 'n_memories': 0, 'memory_strength': 0, 'steps_per_sample': 1, 'gamma': 1.0, 'beta': 1.0, 's': 1, 'batches_per_example': 1, 'bgd_optimizer': 'bgd', 'optimizer_params': '{}', 'train_mc_iters': 5, 'std_init': 0.05, 'mean_eta': 1, 'fisher_gamma': 0.95} # val: 0.455 0.660 0.205 0.259 # 55.097007274627686
'''