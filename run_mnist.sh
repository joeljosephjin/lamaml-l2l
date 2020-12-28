# ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_rotations --cuda --log_dir logs/"
ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_rotations --log_dir logs/"

SEED=0

#lamaml ROTATION MNIST DATASETS
python3 main.py $ROT --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.3 \
                    --alpha_init 0.15 --learn_lr --use_old_task_memory --seed $SEED