import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',     type=int,   default=512,            help='Training batch size')
    parser.add_argument('--learning_rate',  type=float, default=1e-3,           help='Learning Rate')
    parser.add_argument('--beta1',          type=float, default=0.9,            help='Beta1 in Adam')
    parser.add_argument('--beta2',          type=float, default=0.999,          help='Beta2 in Adam')
    parser.add_argument('--epsilon',        type=float, default=1e-8,           help='Epsilon in Adam')
    parser.add_argument('--weight_decay',   type=float, default=1e-4,           help='Weight Decay During Training')
    parser.add_argument('--train_steps',    type=int,   default=10000,          help='The Total Number of Training Steps')
    parser.add_argument('--eval_steps',     type=int,   default=100,            help='The Number of Steps Between Two Evaluations')
    
    parser.add_argument('--split_size',     type=float, default=0.1,            help='The Size to Split Training Set')
    
    parser.add_argument('--load_model',     action='store_true',                help='Whether Load Trained Model')
    parser.add_argument('--do_plot',        action='store_true',                help='Whether Draw Loss / Accuracy Curves')
    parser.add_argument('--plot_steps',     type=int,   default=10,             help='The Number of Steps Between Two Plots')
    
    '''
        TODO
            args.model_name
            args.dataset
            args.num_classes
            args.train_dataset
            args.test_dataset
            ...
    '''
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

args = get_args()