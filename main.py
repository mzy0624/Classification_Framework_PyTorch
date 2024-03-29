from models    import *
from data      import *
from args      import args
from trainer   import ModelTrainer
from evaluator import ModelEvaluator

Model = eval(args.model_name)
model = Model(args.channels, args.num_classes)
model = model.to(args.device)
model_name = f'{model}-{args.dataset}'

print('Load data...')
train_loader, eval_loader, test_loader = get_dataloader(args)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.beta1, args.beta2),
    eps=args.epsilon,
    weight_decay=args.weight_decay
)

evaluator = ModelEvaluator(model, criterion)

if args.load_model:
    print(f'Model {model_name} Loading...')
    model.load_state_dict(torch.load(f'saved_models/{model_name}.bin'))
else:
    print(f'Model {model_name} Training...')
    trainer = ModelTrainer(model, args, evaluator, criterion, optimizer)
    trainer.train(train_loader, eval_loader)

print(f'Model {model_name} Testing...')
evaluator.test(test_loader)