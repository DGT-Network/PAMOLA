import os
import pandas as pd

# Base project to project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to data
DATA_DIR = os.path.join(BASE_DIR, 'data')

# LOAD BASE DATASET FOR MARKET ATTRIBUTION TASK
ma_bank = pd.read_csv(os.path.join(DATA_DIR, 'MA_BANK.csv'))
ma_bank_mask = pd.read_csv(os.path.join(DATA_DIR, 'MA_BANK_MASK.csv'))
ma_ads = pd.read_csv(os.path.join(DATA_DIR, 'MA_ADS.csv'))
ma_ads_mask = pd.read_csv(os.path.join(DATA_DIR, 'MA_ADS_MASK.csv'))

