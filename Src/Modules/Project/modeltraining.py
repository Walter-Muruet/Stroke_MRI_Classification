import numpy as np
import torch
from matplotlib import pyplot as plt
from typing import Callable, Dict, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau

# A set of functions for model training
def train_model(dataloader_train: torch.utils.data.dataloader.DataLoader, 
                dataloader_val: torch.utils.data.dataloader.DataLoader, 
                model: torch.nn.Module, 
                loss_fn: torch.nn.Module, 
                optimiser_fn: torch.optim.Optimizer, 
                lr: float, 
                n_epochs: int, 
                plot_history: bool = True, 
                correct_threshold:float = 0.5, 
                kws_lr_scheduler: Dict[str,float] = {}, kws_early_stop: Dict[str,float] = {}) -> None:
    """
    Abstracts the logic required to train a PyTorch Models. Provides constant status updates on the training process and produces a plot summarising change in accuracy and validation loss across 
    training epochs.
    Params
        dataloader_train, dataloader_val: PyTorch DataLoader objects to load image data.
        model: A PyTorch model.
        loss_fn: A loss function (e.g. Binary Cross Entropy, Categorical Cross Entropy).
        optimiser_fn: An optimisation function (E.g. ADAM, SGD).
        lr: A float. Learning rate.
        n_epochs: An integer. Max number of epochs to train the model.
        plot_history: A bool. Should the function plot accuracy and loss over epochs at the end of the training process?
        correct_threshold: A float. Threshold of probabilities to consider a class labeled as positive. Defaults to 0.5.
        kws_lr_scheduler, kws_early_stop: Optional. Dictionaries with keyword arguments to the learning rate scheduler and the early stop functions. If not passed, this functions are not enabled.
    Returns
        None
    """
    tr_acc_hx, tr_loss_hx = [], []
    vl_acc_hx, vl_loss_hx = [], []

    optimiser = optimiser_fn(model.parameters(), lr = lr)

    if kws_lr_scheduler:
        print('Learning rate annealing enabled...')
        scheduler = ReduceLROnPlateau(optimiser, **kws_lr_scheduler)

    if kws_early_stop:
        print('Early stop enabled...')
        earlyStopper = EarlyStop(**kws_early_stop)

    for epoch in range(n_epochs):
        tr_acc, tr_loss, vl_acc, vl_loss = nn_train_epoch(dataloader_train, dataloader_val, model, loss_fn, optimiser, correct_threshold)
        print(f"Epoch: {epoch + 1};\nTraining accuracy: {tr_acc*100:.2f}%, Training loss: {tr_loss:.5f} ---------- Validation accuracy: {vl_acc*100:.2f}%, Validation loss: {vl_loss:.5f}")

        if kws_lr_scheduler:
            scheduler.step(vl_loss)
            #scheduler.optimizer.param_groups[0]['lr'] # Current learning rate
        
        tr_acc_hx.append(tr_acc)
        tr_loss_hx.append(tr_loss)
        
        vl_acc_hx.append(vl_acc)
        vl_loss_hx.append(vl_loss)

        if kws_early_stop:
            if earlyStopper.early_stop(vl_loss):
                print(f'Early stop at epoch: {epoch + 1} of {n_epochs}')
                break

    if plot_history:
        plt.figure(figsize = (17.5,12))

        # Epochs
        x_axis = np.arange(len(tr_acc_hx)) + 1

        plt.subplot(211)
        plt.title("Loss over epochs")
        plt.plot(x_axis, tr_loss_hx, label = 'Training Loss')
        plt.plot(x_axis, vl_loss_hx, label = 'Validation Loss')
        plt.legend()

        plt.subplot(212)
        plt.title("Accuracy over epochs")
        plt.plot(x_axis, tr_acc_hx, label = 'Training Accuracy')
        plt.plot(x_axis, vl_acc_hx, label = 'Validation Accuracy')
        plt.legend()

#####################################################################################################
def nn_train_epoch(dataloader_train: torch.utils.data.dataloader.DataLoader, 
                   dataloader_val: torch.utils.data.dataloader.DataLoader, 
                   model: torch.nn.Module, 
                   loss_fn: torch.nn.Module, 
                   optimiser: torch.optim.Optimizer, 
                   correct_threshold: float) -> Tuple[float, float, float, float]:
    """
    Abstracts the logic to train one epoch of the model.
    Called by train_model.
    """
    # Training 
    tr_acc, tr_loss = run_epoch(train_batch, dataloader_train, model, loss_fn, optimiser, correct_threshold)

    # Validation
    vl_acc, vl_loss = run_epoch(eval_model, dataloader_val, model, loss_fn, optimiser, correct_threshold)
    
    return tr_acc, tr_loss, vl_acc, vl_loss

#####################################################################################################
def run_epoch(batch_fn: Callable, 
              dataloader: torch.utils.data.dataloader.DataLoader, 
              model: torch.nn.Module, 
              loss_fn: torch.nn.Module, 
              optimser: torch.optim.Optimizer, 
              correct_threshold: float) -> Tuple[float, float]:
    """
    A wrapper function to call train_batch and eval_model.
    Called by nn_train_epoch.
    """
    e_acc, e_loss = [], []
    
    for batch in iter(dataloader):
        X, y = batch
        
        _acc, _loss = batch_fn(X, y.view(-1,1).float(), model, loss_fn, optimser, correct_threshold)
        
        e_acc.extend(_acc)
        e_loss.append(_loss)
    
    return np.mean(e_acc), np.mean(e_loss)

#####################################################################################################
def train_batch(X: torch.tensor,
                y: torch.tensor, 
                model: torch.nn.Module, 
                loss_fn: torch.nn.Module, 
                optimiser: torch.optim.Optimizer, 
                correct_threshold: float) -> Tuple[float, float]:
    """
    Logic to train the model using a single batch of data.
    Called by nn_train_epoch through run_epoch.
    """
    model.train()
    y_pred = model(X)
    
    # Accuracy
    y_correct = ((y_pred > correct_threshold) == y).cpu().numpy().tolist()

    # Loss
    batch_loss = loss_fn(y_pred, y)

    # Backpropagation
    batch_loss.backward()

    # Weights update
    optimiser.step()
    optimiser.zero_grad() # Flush values from the optimiser
    
    return y_correct, batch_loss.item()


#####################################################################################################
@torch.no_grad() # Deactivates the autograd engine, reducing memory usage and increasing speed at the cost of disabling backpropagation
def eval_model(X: torch.tensor,
                y: torch.tensor, 
                model: torch.nn.Module, 
                loss_fn: torch.nn.Module, 
                optimiser: torch.optim.Optimizer, 
                correct_threshold: float) -> Tuple[float, float]:
    """
    Logic to evaluate the model on one batch of data.
    Note that the argument `optimiser` is not used, but is kept for compatibility purposes.
    """
    model.eval() # Set layers in evaluation mode. In particular, layers such as batchnorm and dropout are set to eval mode instead of training mode
    y_pred = model(X)

    # Accuracy
    y_correct = ((y_pred > 0.5) == y).cpu().numpy().tolist()

    # Loss
    val_loss = loss_fn(y_pred, y)
    
    return y_correct, val_loss.item()

#####################################################################################################
def evaluate_model(dataloader: torch.utils.data.dataloader.DataLoader, 
                     model: torch.nn.Module, 
                     loss_fn: torch.nn.Module,  
                     correct_threshold: float) -> Tuple[float, float]:
    """
    Evaluates the performance of a model. It is meant to be used with test data.
    Params:
        dataloader: A PyTorch DataLoader to load test image data.
        model: A PyTorch model.
        loss_fn: A loss function.
        correct_threshold: A float. Threshold of probabilities to consider a class as positive.
    Returns:
        A tuple with two floats (model accuracy and model loss)
    """
    e_acc, e_loss = [], []
    
    for batch in iter(dataloader):
        X, y = batch
        
        _acc, _loss = eval_model(X, y.view(-1,1).float(), model, loss_fn, None, correct_threshold)
        
        e_acc.extend(_acc)
        e_loss.append(_loss)
    
    return np.mean(e_acc), np.mean(e_loss)

####################################################################
# A class to implement Early Stop of model training
class EarlyStop:
    def __init__(self, patience: int, min_delta:float):
        """
        Instantientes a EarlyStop class to prevent overfitting.
        Params:
            patience: An integer. Number of epochs without sufficient improvement before triggering an early stop.
            min_delta: A float. Minimum amount of improvement between two epochs.
        """
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._best_loss = np.inf

    def early_stop(self, val_loss: float) -> bool:
        """
        Evaluates whether validation loss is continuously improving each epoch by a certain threshold.
        Params:
            val_loss: A float representing the validation loss at the current epoch.
        """
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._counter = 0
        elif val_loss > (self._best_loss + self._min_delta):
            self._counter += 1
            if self._counter >= self._patience:
                return True
        return False