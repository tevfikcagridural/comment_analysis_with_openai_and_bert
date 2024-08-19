from typing import Tuple
from utils.bgg_api_client import BGGClient
from utils.comment_cleaning import clean_comments
from utils.aspect_extraction import aspect_extraction_for_game
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import os
import sys
sys.path.append('../')

## Download and prepare data
def prepapre_data(top_n: int | None = 2) -> None:
    """
    Main function to download and process comments of top n games from Board Game Geek API.
    
    Args:
    
    top_n : Optional[int]
        The number of top ranked games for which comments should be downloaded and processed, defaults to 2 (top 2 games). If set as None it will process up to the top 100 games.
    
    Returns:
        None
        
    """
    
    # Initialize the client and download comments of top n games
    bgg_client = BGGClient()
    
    # Download top 100 game's information
    # Commented out since already downloaded in the notebook
    # bgg_client.get_games() 
    
    if top_n == None:
        top_n = 100
        
    games_df = pd.read_csv('data/raw/top_100_games.csv')[:top_n]
    
    # Download, clean and extract aspects for the top n games
    # Be careful that higher numbers will cause OpenAI uasage costs
    for idx, row in games_df.iterrows():
        
        game_name = row['TITLE'][0]
        # Download comments
        bgg_client.download_comments(game_name)

        # Remove non-English comments
        clean_comments(game_name)

        # Get aspects for each comment of all games
        aspect_extraction_for_game(game_name)


## Setup data for model development
NUM_WORKERS = 0
MAX_LEN = 200
BATCH_SIZE = 8
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


## Reference: transformers_multi_label_classification.ipynb
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.COMMENT
        self.targets = self.data.ENCODE
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int=NUM_WORKERS,
    is_aspect: bool=True
) -> Tuple:
    """Creates training adn desting DataLoaders.
    
    Takes in a data directory of multiple CSV files, reads and combines them
    into PyTorch DataLoaders

    Args:
        data_dir (str): Directory of CSV files of each game's comment and labels
        batch_size (int): Number of samples per batch in each of the DataLoaders.
        num_workers (int, optional): n integer for number of workers per DataLoader. Defaults to available cpu count.
        is_aspect (bool, optional): If the model will be trained on aspects or categories
    Returns:
        A tuple of (train_dataloader, test_dataloader, encoded_list).
    """
    files = [data_dir+file for file in os.listdir(data_dir) if file.endswith('.csv')]
    df = pd.DataFrame()
    for file in files:
        df_temp = pd.read_csv(file)
        df = pd.concat([df, df_temp])
    
    def idx_extractor(string, labels):
        """Helper function to convert LLM returned list into one hot encoding list"""
        encoded = []
        for idx, l in enumerate(labels):
            search = string.find(l)
            if search == -1:
                encoded.append(0)
            else:
                encoded.append(1)
        return encoded
    
    # Create OHE lists for aspects or categories
    if is_aspect:
        aspects = ['LUCK', 'BOOKKEEPING', 'DOWNTIME', 'INTERACTION', 'BASH THE LEADER']
        encodes = df.LABELS.apply(lambda x: idx_extractor(x, aspects))
    else:
        categories = ['COMPLICATED', 'COMPLEX']
        encodes = df.LABELS.apply(lambda x: idx_extractor(x, categories))
        
    df['ENCODE'] = encodes
    
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=42)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
    
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': num_workers,
                }

    test_params = {'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': num_workers,
                    }

    train_loader = DataLoader(training_set, **train_params)
    test_loader = DataLoader(testing_set, **test_params)
    
    return train_loader, test_loader, encodes