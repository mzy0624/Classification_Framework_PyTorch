import torch

class ModelEvaluator:
    def __init__(self, model, criterion):
        self.model       = model
        self.criterion   = criterion
   
    def evaluate(self, eval_loader):
        self.model.eval()
        eval_acc = [0, 0]
        with torch.no_grad():
            for X, y in eval_loader:
                preds = self.model(X).argmax(dim=-1)
                eval_acc[0] += (preds == y).sum()
                eval_acc[1] += len(preds)
        return eval_acc[0] / eval_acc[1]
        
    def test(self, test_loader):
        test_acc = self.evaluate(test_loader)
        print(f'Test accuracy: {test_acc * 100:.2f}%')