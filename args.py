import argparse
import torch

parser = argparse.ArgumentParser()
# Custom Arguments
parser.add_argument('--seed',               type=int,   default=3407,               help='Random seed')
parser.add_argument('--gpu_id',             type=str,   default='0',                help='The GPU ID')
parser.add_argument('--mode',               type=str,   default='train',            help='Choose mode from "train", "test" and "distill"')

# Dataset Argument
parser.add_argument('--dataset',            type=str,   default='CIFAR100',         help='The dataset')

# Training Argument
parser.add_argument('--model',              type=str,   default='ResNet18',         help='The model name')
parser.add_argument('--batch_size',         type=int,   default=512,                help='Training batch size')
parser.add_argument('--epochs',             type=int,   default=30,                 help='The number of training epochs')
parser.add_argument('--lr',                 type=float, default=0.01,               help='Learning rate')
parser.add_argument('--weight_decay',       type=float, default=1e-4,               help='Weight decay rate')
parser.add_argument('--momentum',           type=float, default=0.9,                help='Momentum in SGD')
parser.add_argument('--no_saving',          action='store_true',                    help='Whether save model during training process')

# Testing Argument
parser.add_argument('--test_models',        type=str,   nargs='+',                  help='Models to be tested')

# Ploting Arguments
parser.add_argument('--plot_steps',         type=int,   default=10,                 help='The number of steps between two plots')
parser.add_argument('--no_plot',            action='store_true',                    help='Training without drawing loss / accuracy curves')

args = parser.parse_args()

torch.manual_seed(args.seed)
args.device = torch.device('cuda:' + args.gpu_id if torch.cuda.is_available() else 'cpu')
args.num_classes = int(args.dataset[5:])    # CIFAR10: 10, CIFAR100: 100