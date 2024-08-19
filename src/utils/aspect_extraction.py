from utils.log_handling import log_handler 
import pandas as pd
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import sys
sys.path.append('../')

load_dotenv()    

openai.api_key = os.getenv('OPENAI_API_KEY')

def _get_aspects_for_comment(comment, model="gpt-3.5-turbo") -> str | None:
    """
    This function uses OpenAI's API to extract the aspects-categories from a given comment.
    
    It prompts OpenAI with the system prompt and the comment provided, then returns the extracted aspect-category as a string of lists.
    
    Args:
        - comment: The text comment for which we want to extract its aspect.
        - model: The OpenAI model to use. Default is "gpt-3.5-turbo".
    
    Returns:
        - The extracted aspects and categories as a string, or None if the extraction fails.
    """
    # Build message for the model
    with open('../config/aspect_extractor.prompt', 'r') as f:
        system_prompt = f.read()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": comment}
    ]
    
    # Send request to the model
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content if response.choices else None

def aspect_extraction_for_game(game_name: str, model="gpt-3.5-turbo", logger=log_handler('aspect_extraction'), ) -> None:
    """
    This function reads cleaned comments for a given game and uses OpenAI API to extract aspects from each comment. 
    
    It then saves the result as a new CSV file containing both original comments and extracted aspect-categories.
    
    Args:
        - game_name: The name of the game, used to determine the path for reading cleaned comments and writing results.
        - logger: Optional logging handler with default value log_handler('aspect_extraction'). Logging any errors that occur during extraction process.
        - model: The OpenAI model to use. Default is "gpt-3.5-turbo".
    
    Returns:
        None
        
    """
    # Read cleaned comments for the given game
    cleaned_csv_folder_path = 'data/interim/games'
    df = pd.read_csv(cleaned_csv_folder_path + game_name + '.csv', usecols=['COMMENT'])
    
    try:
        # Loop through each row of the DataFrame, extracting aspects using OpenAI API
        for idx, row in df.iterrows():
            df.loc[[idx],'ASPECTS'] = _get_aspects_for_comment(row['COMMENT'], model)
            
            
        # Save new dataframe to CSV file
        df.to_csv('data/processed/{game}_comments_with_aspects.csv', index=False)
        
        # Log successful aspect extraction for the game
        logger.info(f"Aspect extraction completed for game: {game_name}")
    
    except Exception as e:
        # Log any errors that occur during aspect extraction process
        logger.error(f"Error during aspect extraction for game {game_name}: {e}")
