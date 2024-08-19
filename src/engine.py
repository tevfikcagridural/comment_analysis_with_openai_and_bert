import logging
import torch
from typing import Dict, List, Tuple

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,   # type: ignore
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Performs a single training step.
    
    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader instance containing the training data.
        loss_fn (torch.nn.Module): Loss function used during training.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters.
        device (torch.device): Device on which the model and data are located.
        
    Returns:
        Tuple[float, float]: A tuple containing the total training loss and the loss from this step.
    """
    
    model.train()
    
    train_loss = 0
    
    # Loop throgh data loader with batches
    for _, data in enumerate(dataloader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        y = data['targets'].to(device)
        
        y_pred = model(ids, mask, token_type_ids)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Adjust metric to get average
    train_loss /= len(dataloader)
    
    return train_loss, loss.item()

def test_step( 
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, #type: ignore 
    loss_fn: torch.nn.Module, 
    device: torch.device 
) -> Tuple[float, float]: 
    """
    Performs a single testing step.
    
    This function is used to evaluate the performance of a model on a test dataset.
    It's similar to train_step(), but it doesn't perform any optimization or gradient calculations.

    Args:
        model (torch.nn.Module): The model being tested.
        dataloader (torch.utils.data.DataLoader): The data loader for the test set.
        loss_fn (torch.nn.Module): The loss function used to calculate the loss between predicted and actual outputs.
        device (torch.device): The device on which the computation is performed.
        
    Returns:
        Tuple[float, float]: A tuple containing the total loss and the average loss per batch.
    """ 
    test_loss = 0 
    
    with torch.no_grad(): 
        for _, data in enumerate(dataloader, 0): 
            ids = data['ids'].to(device, dtype=torch.long) 
            mask = data['mask'].to(device, dtype=torch.long) 
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long) 
            y = data['targets'].to(device) 
            
            y_pred = model(ids, mask, token_type_ids) 
            loss = loss_fn(y_pred, y) 
            test_loss += loss.item() 
            
    # Adjust metric to get average
    test_loss /= len(dataloader) 
    
    return test_loss, loss.item()

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader, #type: ignore   
    test_dataloader: torch.utils.data.DataLoader, #type: ignore 
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    logger: logging.Logger
) -> dict:
    """
    Trains and tests the model.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_fn (torch.nn.Module): Loss function used for training.
        epochs (int): Number of training epochs.
        device (torch.device): Device to use for training and testing.
        logger (logging.Logger): Logger to log training metrics.
        
    Returns:
        dict: A dictionary containing the history of training and testing losses.
    """
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    for epoch in range(epochs):
        # Training step
        train_loss, loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Testing step
        test_loss, loss = test_step(model, test_dataloader, loss_fn, device)
        history['test_loss'].append(test_loss)
        
        # log metrics
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        
    return history