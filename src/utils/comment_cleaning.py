from py3langid.langid import LanguageIdentifier, MODEL_FILE
import pandas as pd
import sys
from utils.log_handling import log_handler

sys.path.append('../')
logger = log_handler('comment_cleaning')

# Initialize language identifier
identifier = LanguageIdentifier.from_pickled_model(
    MODEL_FILE, 
    norm_probs=True # Normalize the probabilities of the model
)

def clean_comments(game_name: str) -> None:
    """
    This function cleans comments from a given dataset. The dataset is expected to be in .csv format.
    
    Args:
        game_name: The name of the csv file containing comment data, including its extension.
                      For example, 'game1.csv' or just 'game1'. 
    Returns:
        None - this function only saves a cleaned DataFrame back to .csv format under the directory 
             'data/interim', with original filename as game_name parameter value.
    """
    
    try:
        df = pd.read_csv(game_name, header=None)
        df.rename({0: 'COMMENT'}, axis=1, inplace=True)
        
        # Apply the identification to the DataFrame
        df[['LANGUAGE', 'PROBABILITY']] = df['COMMENT'].apply(lambda x: pd.Series(identifier.classify(x)))

        # Remove non-English comments
        df = df[df['LANGUAGE'].isin(['en'])]

        df.to_csv(f'data/interim/games/{game_name}', index=False)
        logger.info(f"non-English comments removed for the game: {game_name}")
    except Exception as e:
        logger.error(f"Error: {e}")