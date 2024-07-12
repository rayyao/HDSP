import os
import argparse
import random
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str,  default='', help='training script name')
    parser.add_argument('--config', type=str,default='', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, default='', help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, default=torch.cuda.device_count(), help="number of GPUs per node")
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb

    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print('args.config ', args.config)
    if args.mode == "single":
        train_cmd = ""\
                    % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                       args.distill, args.script_teacher, args.config_teacher, args.use_wandb)
    elif args.mode == "multiple":
        train_cmd = "" \
                    "" \
                    % (args.nproc_per_node, random.randint(10000, 50000), args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    elif args.mode == "multi_node":
        train_cmd = "" \
                    "" \
                    % (args.nproc_per_node, args.ip, args.port, args.world_size, args.rank, args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    else:
        raise ValueError("mode should be 'single' or 'multiple' or 'multi_node'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
