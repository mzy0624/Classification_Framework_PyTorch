import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

def get_transforms(train=True):
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4468), 
            (0.2675, 0.2565, 0.2761), 
            inplace=True
        )
    ]
    train_transform_list = [
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip()
    ] if train else []
    return transforms.Compose(train_transform_list + transform_list)

def get_dataloaders(args):
    Dataset = eval(args.dataset)
    trainset = Dataset(root=f'dataset/{args.dataset}', train=True,  transform=get_transforms(),            download=True)
    testset  = Dataset(root=f'dataset/{args.dataset}', train=False, transform=get_transforms(train=False), download=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=3, pin_memory=True, shuffle=True)
    testloader  = DataLoader(testset,  batch_size=args.batch_size, num_workers=3, pin_memory=True,)    
    return trainloader, testloader
