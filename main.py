import torch
import torch.optim  as optim
from args           import args
from data           import get_dataloaders
from models         import get_model
from ModelTrainer   import ModelTrainer
from ModelEvaluator import ModelEvaluator

if __name__ == '__main__':
    print(f'Load {args.dataset} dataset...')
    train_loader, test_loader = get_dataloaders(args)
    print(f'Dataset {args.dataset} loaded.')

    if args.mode == 'train':
        model, model_name = get_model(args.model, args)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        scheduled = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, 
            max_lr=args.lr, 
            epochs=args.epochs, 
            steps_per_epoch=len(train_loader)
        )
        print(f'Model {model_name} training...')
        trainer = ModelTrainer(model, args, train_loader, test_loader, optimizer, scheduled)
        trainer.optimize()
    
    elif args.mode == 'test':
        for test_model in args.test_models:
            model, model_name = get_model(test_model, args, from_pretrained=True)            
            evaluator = ModelEvaluator(model)
            print(f'Model {model_name} testing...')
            evaluator.test(test_loader)
    
    else:
        assert 0, f"Mode {args.mode} Invalid."