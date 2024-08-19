import torch
import data_setup, engine, model
from multiprocessing import freeze_support
from utils.log_handling import log_handler
import sys
sys.path.append('../')

if __name__ == '__main__':
    freeze_support()
    
    # Setup hyperparams
    NUM_EPOCHS = 2
    BATCH_SIZE = 8
    LEARNING_RATE = 1.0e-5

    # Setup data directory, model name, and logging 
    data_dir = 'data/processed/random_game_comments/'
    model_name = 'random_100k_comments'
    logger = log_handler(f"{model_name} train_test process")
    logger.info(f"Hyperparameters: Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE} | Learning Rate: {LEARNING_RATE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create DataLoaders
    train_dataloader, test_dataloader, encode_list = data_setup.create_dataloaders (
        data_dir=data_dir,
        batch_size=BATCH_SIZE
    )

    # Create Model
    model = model.BERTClass().to(device)

    # Set loss function and optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


    # Begin training
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        logger=logger,
        device=device,
    )

    torch.save(model.state_dict(), 'models/' + model_name + '.pth')
