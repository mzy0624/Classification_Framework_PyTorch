import torch

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
    
    def accuracy(self, output, target):
        with torch.no_grad():
            preds = output.argmax(dim=-1)
            correct = (preds == target).sum()
            total = len(target)
        return correct, total
        
    def evaluate(self, eval_loader):
        self.model.eval()
        eval_acc = [0, 0]
        with torch.no_grad():
            for X, y in eval_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                output = self.model(X)
                correct, total = self.accuracy(output, y)
                eval_acc[0] += correct
                eval_acc[1] += total
        return eval_acc[0] / eval_acc[1] * 100
        
    def test(self, test_loader):
        test_acc = self.evaluate(test_loader)
        print(f'Test accuracy: {test_acc:.2f}%')