import torch, numpy

# implementation of early stopping strategy
class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
            Args:
                patience (int): How long to wait after last time validation loss improved.
                                Default: 7
                verbose (bool): If True, prints a message for each validation loss improvement.
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = numpy.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score - self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}, valid error is {score:.8f} vs {self.best_score:.8f}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(torch.load('checkpoint.pt'))
                print(f'valid error is {score:.8f} vs {self.best_score:.8f}')   
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
            Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.9f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')     # save the parameter of the best model
        torch.save(model, 'finish_model.pkl')                 # save the best model
        self.val_loss_min = val_loss