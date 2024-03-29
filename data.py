from pandas import read_csv
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import random

class CustomDataset(Dataset):
    def __init__(self, dataset):
        super(CustomDataset, self).__init__()
        self.features = []  # reshape and float
        self.labels = []

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def split_train_eval(dataset, eval_size=0.1):
    num_examples = len(dataset)
    indices = range(num_examples)
    num_evals = int(num_examples * eval_size)
    eval_idx = random.sample(indices, k=num_evals)
    train_idx = list(set(indices) - set(eval_idx))
    return dataset[train_idx], dataset[eval_idx]
    
def get_MNIST(filename):
    df = read_csv(open(filename), header=None)
    dataset = torch.tensor(df.values) 
    return dataset      # N * (1 + 28 * 28) = N * 785

def get_dataset(args):
    dataset_path    = f'dataset/{args.dataset}/'
    trainset_path   = dataset_path + args.train_dataset
    testset_path    = dataset_path + args.test_dataset
    data_func       = eval(f'get_{args.dataset}')       # TODO: load data from file. Example: MNIST
    train_set       = data_func(trainset_path).to(args.device)
    test_set        = data_func(testset_path).to(args.device)
    train_set, eval_set = split_train_eval(train_set, args.split_size)
    return train_set, eval_set, test_set

def get_dataloader(args):
    train_set, eval_set, test_set = get_dataset(args)
    train_loader = DataLoader(CustomDataset(train_set, args.shape), batch_size=args.batch_size, shuffle=True)
    eval_loader  = DataLoader(CustomDataset(eval_set,  args.shape), batch_size=args.batch_size)
    test_loader  = DataLoader(CustomDataset(test_set,  args.shape), batch_size=args.batch_size)
    train_loader = cycle(train_loader)
    return train_loader, eval_loader, test_loader