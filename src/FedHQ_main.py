import os
from options import args_parser
from train import train
args = args_parser()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    train()