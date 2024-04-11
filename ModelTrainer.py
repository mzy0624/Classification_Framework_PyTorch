import torch
from torch.nn.functional import cross_entropy
from time import time
from tqdm import tqdm
from Visdom import Visdom
from ModelEvaluator import ModelEvaluator

class ModelTrainer:
    def __init__(self, model, args, train_loader, eval_loader, optimizer, scheduled=None):
        self.model        = model
        self.args         = args
        self.train_loader = train_loader
        self.eval_loader  = eval_loader
        self.optimizer    = optimizer
        self.scheduled    = scheduled
        self.evaluator    = ModelEvaluator(model)
        self.steps        = 0
        self.loss         = 0.0
        self.acc          = [0, 0]  # [correct, total]: acc = correct / total
        self.max_acc      = 0.0

    def optimize(self):
        torch.cuda.empty_cache()
        self.vis = None if self.args.no_plot else Visdom(str(self.model))
        self.time = time()
        for epoch in range(self.args.epochs):
            self.optimize_single_epoch(epoch + 1)
    
    def optimize_single_epoch(self, epoch):
        self.model.train()
        bar = tqdm(self.train_loader)
        bar.set_description(f'Train epoch: [{epoch} / {self.args.epochs}]')
        
        for step, (X, y) in enumerate(bar):
            loss = self.optimize_single_step(X, y, step)
            bar.set_postfix(loss=f'{loss:.2f}')
        
        eval_acc = self.evaluator.evaluate(self.eval_loader)
        if self.max_acc < eval_acc:
            self.max_acc = eval_acc
            if not self.args.no_saving:
                self.model_saving()
        if not self.args.no_plot:
            self.vis.plot(epoch, eval_acc, 'Evaluation', name=f'Eval-acc')
        print(f'Eval acc: {eval_acc:.2f}% | max acc: {self.max_acc:.2f}% | Time: {time() - self.time:.3f}s')
        
    def optimize_single_step(self, X, y, step):
        self.steps += 1
        X = X.to(self.args.device)
        y = y.to(self.args.device)
        out = self.model(X)
        loss = cross_entropy(out, y)
                
        batch_acc = self.evaluator.accuracy(out, y)
        self.loss += loss
        self.acc[0] += batch_acc[0]
        self.acc[1] += batch_acc[1]
        avg_loss = self.loss / self.steps
        avg_acc = self.acc[0] / self.acc[1] * 100
        if not self.args.no_plot and step % self.args.plot_steps == 0:
            self.vis.plot(
                self.steps / len(self.train_loader), avg_loss, 'Training', 
                title='Loss', xlabel='epoch', ylabel='loss'
            )
            self.vis.plot(
                self.steps / len(self.train_loader), avg_acc, 'Evaluation', name='Train-acc', 
                legend=['Train-acc', 'Eval-acc'], title='Accuracy', xlabel='epoch', ylabel='accuracy'
            )

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduled is not None:
            self.scheduled.step()

        return avg_loss
        
    def model_saving(self):
        torch.save(self.model.state_dict(), f'saved_models/{self.model}-{self.args.dataset}.bin')
        print(f'Model {self.model} saved.')