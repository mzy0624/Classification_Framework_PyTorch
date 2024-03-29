import torch
from torch.nn.functional import one_hot
from tqdm import trange
from Visdom import Visdom
        
class ModelTrainer:
    def __init__(self, model, args, evaluator, criterion, optimizer):
        self.model        = model
        self.args         = args
        self.evaluator    = evaluator
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.max_acc      = 0.0
    
    def train(self, train_loader, eval_loader):
        if self.args.do_plot:
            vis = Visdom(str(self.model))
        bar = trange(self.args.train_steps)
        train_loss = 0.0
        for step in bar:
            self.model.train()
            X, y = next(train_loader)
            y = one_hot(y, num_classes=self.args.num_classes).float()
            scores = self.model(X)
            loss = self.criterion(scores, y)
            train_loss += loss
            bar.set_postfix(loss=f'{loss:.3f}')
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.args.do_plot and step % self.args.plot_steps == 0:
                vis.plot(
                    step + 1, train_loss / (step + 1), 'train', 
                    legend=str(self.model), title='Training loss', 
                    xlabel='train batches', ylabel='training loss'
                )
                
            if (step + 1) % self.args.eval_steps == 0:
                eval_acc = self.evaluator.evaluate(eval_loader)
                if eval_acc > self.max_acc:
                    self.max_acc = eval_acc
                    self.model_saving()
                print(f'eval acc = {eval_acc * 100:.2f}% | max eval acc = {self.max_acc * 100:.2f}%')
                if self.args.do_plot:
                    vis.plot(
                        step + 1, eval_acc, 'eval',
                        legend=str(self.model) + '_acc', title='Evaluation',
                        xlabel='eval_batches', ylabel='Accuracy'
                    )

    def model_saving(self):
        torch.save(self.model.state_dict(), f'saved_models/{self.model}-{self.args.dataset}.bin')
        print(f'Model saved. Eval acc: {self.max_acc * 100:.2f}%')