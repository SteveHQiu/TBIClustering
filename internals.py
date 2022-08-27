#%% Imports
from typing import Union
import os, csv, sys
from datetime import datetime

import pandas as pd
from pandas import DataFrame

# Probably should not have internal imports for global_functions to avoid circular imports 
#%% Logging 
import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG) # Sets level at which info will be captured, can elevate level for CLI and file output to filter out lower level messages
formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S")

# Logging CLI output stream
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO) # Set to INFO to display only up to INFO level
ch.setFormatter(formatter)
LOG.addHandler(ch)

# Logging file output stream
date_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
if os.path.exists("archive/logs/"):
    log_dir = "archive/logs/"
else:
    log_dir = "" # Use root dir
# fh = logging.FileHandler(F"{log_dir}{date_time}.log", "w")
fh = logging.FileHandler(F"{log_dir}{date_time}.log", "w")
fh.setLevel(logging.DEBUG) # Log info all the way down to DEBUG level  
fh.setFormatter(formatter)
LOG.addHandler(fh)
#%% Functions 

def mergeDfSlices(prefix: str, dir: str = os.getcwd()):
    """
    Merges Excel and CSV of a certain root prefix in a given directory into one Excel file in the same directory

    Args:
        prefix (str): Prefix of filenames to be merged, will be used to search 
        dir (str, optional): Relative path to directory containing files to be merged Defaults to os.getcwd().
    """
    root, dirs, files = list(os.walk(dir))[0] # os.walk yields generator that should only get one item, use index 0 to obtain it
    df_merged = pd.DataFrame()
    for file_name in files:
        if file_name.startswith(prefix) and file_name.endswith((".xlsx", ".xls", ".csv")):
            df = importData(os.path.join(dir, file_name)) # Has support for both XLS and CSV
            df_merged = pd.concat([df_merged, df])

    df_merged.to_excel(os.path.join(dir, f"{prefix}_merged.xlsx"))

def importData(file_path: Union[str, bytes, os.PathLike],
               cols: list[str] = [],
               screen_dupl: list[str] = [],
               screen_text: list[str] = [],
               filt_col: str = "",
               filt: str = "",
               skiprows: int = 0,
               ) -> DataFrame:
    """
    Returns entire processed DF based on imported Excel data filterd using preliminary str filter
    If 
    
    file_path: Filepath to Excel file containing data
    cols: Labels of columns to import, will import all if empty 
    screen_dupl: list of columns to check for duplicates between rows, does not combine the list items like built-in behaviour but rather iterates through each
    screen_text: list of columns to check for presence of text, does not combine the list items like built-in behaviour but rather iterates through each
    filt: REGEX string to filter cell
    filt_col: String of column to apply filter to
    skiprows: number of rows to skip when processing data
    """
    # Import function 
    if (file_path.endswith(".xls") or file_path.endswith(".xlsx")):
        df = pd.read_excel(file_path, skiprows = skiprows)
    elif (file_path.endswith(".csv")):
        df = pd.read_csv(file_path, skiprows = skiprows)
    elif (file_path == ""):
        LOG.warning("Empty file path, returning empty DataFrame")
        return DataFrame() # Return empty dataframe to maintain type consistency
    else:
        LOG.warning("Invalid filetype, returning empty DataFrame")
        return DataFrame() # Return empty dataframe to maintain type consistency
    
    # Extra pre-processing functions
    if screen_dupl:
        for col in screen_dupl: # Drop duplicates for every column mentioned, built-in behaviour is to look at combination of columns: https://stackoverflow.com/questions/23667369/drop-all-duplicate-rows-across-multiple-columns-in-python-pandas
            df = df.drop_duplicates(subset=[col])
    if screen_text:
        for col in screen_text: # Go through every screen_text col and check that it has a non-empty string
            df = df.dropna(subset=[col]) 
            df = df[df[col].str.contains(r"[A-Za-z]", regex=True) == True] # Only allow non-empty strings through
    if filt_col and filt: # If both fields are not empty
        df = df[df[filt_col].str.contains(r"[A-Za-z]", regex=True) == True] # Only allow non-empty strings through
        df = df.loc[df[filt_col].str.contains(filt, regex=True, case=False) == True] # Filters abstracts based on a str using regex, regex searches probably searches .lower() str of cell for case insensitivity
    if screen_dupl or screen_text or (filt_col and filt): # Only reset index if one of these previous operations has been done
        df: DataFrame = df.reset_index(drop=True) # to re-index dataframe so it becomes iterable again, drop variable to avoid old index being added as a column
    if cols: # If cols is not empty, will filter df through cols, otherwise leave df unchanged
        df = df[cols] 
        
    return df

def exportJSON(obj, path):
    return

def importJSON(path):
    return 