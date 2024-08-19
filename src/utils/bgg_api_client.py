from utils.log_handling import log_handler
from typing import List
import pandas as pd
import requests
import xml.etree.ElementTree as ET        
import re
import time
import csv
import os
import sys
sys.path.append('../')


class BGGClient:
    def __init__(
            self, 
            api_url: str='https://www.boardgamegeek.com/xmlapi2/', 
            page_size: int=100, 
            max_retries: int =5,
            min_comment_length: int = 200
        ) -> None:
        """Instantiate a new BGGClient with the given api_url, page_size and max_retries.
        
        Parameters:
        -----------
        api_url : str
            
            URL of the xml api to use when making requests. 
            Defaults to 'https://www.boardgamegeek.com/xmlapi2/'
        page_size : int
        
            The number of items in each response from a request. 
            Defaults to 100.
            
        max_retries : int
        
            The maximum number of retries to make when the API returns a non-200 status code. 
            Defaults to 5.
            
        min_comment_length: str
        
            Bottom limit for the comment to download. Comments under this number of characters will not be downloaded
    
        Returns:
        --------
        A new instance of BGGClient with the given parameters

        Example:
        --------
        client = BGGClient(api_url='https://www.boardgamegeek.com/xmlapi2')
        
            Instantiates a new instance of BGGClient using https://www.boardgamegeek.com/xmlapi2 as the api url.
            
        """
        self._api_url = api_url
        self._page_size = page_size
        self._log_name = 'boardgamegeek_api_logger'
        self._logger = log_handler(self._log_name)
        self._max_retries = max_retries
        self.min_comment_length = min_comment_length
        
    def get_games(self) -> None:
        """Retrieve the top 100 board games from BoardGameGeek, as ranked by number of votes. 
        These games will then be saved as top_100_games in the foder data/raw/
        
        Returns:
            Nothing
        """
        
        # Retrieve top 100 board games
        df = pd.read_html(
            'https://boardgamegeek.com/browse/boardgame?sort=rank&rankobjecttype=subtype', 
            extract_links="body" # To get game id's from thumbnail image address
            )[0]
        
        # Extract info from tuples within rows
        df = df.assign(
            BOARD_GAME_RANK = lambda x: x['Board Game Rank'].apply(lambda y: y[0]),
            THUMBNAIL_ADDRESS = lambda x: x['Thumbnail image'].apply(lambda y: y[1]),
            TITLE = lambda x: x['Title'].apply(lambda y: y[0]),
            GEEK_RATING = lambda x: x['Geek Rating'].apply(lambda y: y[0]),
            AVG_RATING = lambda x: x['Avg Rating'].apply(lambda y: y[0]),
            NUM_VOTERS = lambda x: x['Num Voters'].apply(lambda y: y[0]),
        ).iloc[:,-6:] # Remove original columns-

        # Create a search pattern to extract game id from 
        pattern = re.compile(r'(?<=boardgame/)[0-9]*')
        mask = df['THUMBNAIL_ADDRESS'].apply(
            lambda x: pattern.findall(str(x))
        )
        
        # Remove rows without id
        df = df[~mask.apply(len).eq(0)]
        
        # Set GAME_ID 
        id_series = mask[~mask.apply(len).eq(0)]
        df['GAME_ID'] = id_series.apply(lambda x: int(x[0]))
        
        # Title correction
        df['TITLE'] = df['THUMBNAIL_ADDRESS'].apply(
            lambda x: str(x).split('/')[-1].replace('-', '_')
        )
        
        # Store the df for fetching comments
        df.to_csv('data/raw/top_100_games.csv', index=False)
    
    def _get_comments_for_game(self, game_id: int, page: int = 1) -> List[str]:
        """Fetches comments for a game with the given ID.

        Args:
            game_id (int): The ID of the game.
            logger: A logging object to log messages.
            page (int, optional): The page number. Defaults to 1.
            page_size (int, optional): The number of comments per page. Defaults to 100.
            max_retries (int, optional): The maximum number of retries if an error occurs. Defaults to 5.

        Returns:
            List[str]: A list of comments for the game. If an error occurred, it returns SystemError.
        """
        url = self._api_url + f'thing?id={game_id}&comments=1&page={page}&pagesize={self._page_size}'
    
        retries = 0
    
        def back_off(retries: int, error) -> None:
            self._logger.warning(f"Backing off for: {error}")
            retries += 1
            wait_time = 2 ** retries
            self._logger.info(f"Retrying in {wait_time} seconds")
            self._logger.info(f"Try num: {retries}")
            time.sleep(wait_time)

        while retries < self._max_retries:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # Parse the XML response
                    root = ET.fromstring(response.content)
                    page_comments = []
                    comments = root.findall(".//comment")
                    if len(comments) == 0:
                        self._logger.info(f"Processed completed for the game. Moving on to next one.")
                        return page_comments
                    for comment in comments:
                        comment_value = comment.attrib.get('value')
                        # Saving comments with more chars from lower limit
                        if len(comment_value) >= self.min_comment_length: # type: ignore
                            page_comments.append(comment_value)
                        else:
                            pass
                    self._logger.info(f"Pulled {len(page_comments)} comments from page {page}")
                    return page_comments
                elif response.status_code == 429:
                    back_off(retries, error=429)
            except requests.exceptions.HTTPError as e:
                back_off(retries, e)
            except requests.exceptions.ChunkedEncodingError as e:
                back_off(retries, e)
        
        self._logger.error(f"Failed to fetch comments for game {game_id} (page {page}) after {self._max_retries} attempts")
        raise SystemError
    
    def _save_comments(self, game_name, logger, comments) -> None:
        """Saves given list of comments to csv file named with game's name.

        Args:
            game_name (str): Name of the game
            comments (list): List of comments to be saved
        """
        standard_game_name = game_name.replace(':', '').replace(' ', '_')
        csv_filename = f"data/raw/games/{standard_game_name}.csv"
        
        # Create if game CSV file doesn't exist
        if not os.path.exists(csv_filename):
            write_status = 'w'
        else: 
            write_status = 'a'
            
        with open(csv_filename, write_status, newline='') as csvfile:
            writer = csv.writer(csvfile)
            for comment in comments:
                writer.writerow([comment])
            if len(comments) >= 1:
                logger.info("Comments saved")
    
    def download_comments(self, game_name: str) -> None:
        """
        Downloads and saves the comments for a given game.
        
        Args:
            game_name (str): The name of the game.
        
        Returns:
            None
        """
        try:
            df = pd.read_csv('/data/raw/top_100_games.csv')
            
            self._logger.info(f"Processing for game {game_name}")
            
            game_id = df[df['TITLE'] == game_name, 'GAME_ID'][0]
            # Expecting there wouldn't be a game with 100'000 comments 
            for page_num in range(1, 1000):
                comments = self._get_comments_for_game(game_id, page_num)
                # When the final page reached server returns an empty list
                # Therefore we know we're done with this game
                if len(comments) == 0:
                    break
                self._save_comments(game_name, self._logger, comments)
                
            self._logger.info(f"All comments saved")
        
        except FileNotFoundError as e:
            print("Game information CSVs are not found. Run `get_games` method first for top 100 games")
