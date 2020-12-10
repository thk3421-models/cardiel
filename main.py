import argparse
import datetime
import pandas as pd
import sys
import json
import yfinance
from typing import Dict

def load_config(path:str)->Dict:
    "load required config file"
    with open(path) as config_file:
        data = json.load(config_file)
    return data

def load_prices(config:Dict)->pd.DataFrame:
    "load prices from web or from local file"
    if OPTIONS.price_data == 'yahoo':
        stock_symbols, crypto_symbols = [], []
        start_date = (datetime.datetime.today()
                      - datetime.timedelta(days=365*config['max_lookback_years'])).date()
        end_date = datetime.datetime.today().date() - datetime.timedelta(days=1)
        try:
            if 'symbols' in config.keys():
                stock_symbols = config['symbols']
            symbols = sorted(stock_symbols)
        except KeyError:
            print('Error retrieving symbols from config file. Config file should be \
                   formatted in JSON such that config[\'assets\'][\'stock_symbols\'] \
                   is valid. See example config file from GitHub')
            sys.exit(-1)
        if len(symbols) > 0:
            print('Downloading adjusted daily close data from Yahoo! Finance')
            try:
                price_data = yfinance.download(symbols, start=str(start_date), end=str(end_date),
                                               interval='1d', auto_adjust=True, threads=True)
            except:
                print('Error downloading data from Yahoo! Finance')
                sys.exit(-1)
            cols = [('Close', x) for x in symbols]
            price_data = price_data[cols]
            price_data.columns = price_data.columns.get_level_values(1)
            price_data.to_csv('sample_data.csv', header=True)
    elif OPTIONS.price_data is not None:
        try:
            #Expects a CSV with Date, Symbol header for the prices, i.e. Date, AAPL, GOOGL
            price_data = pd.read_csv(OPTIONS.price_data, parse_dates=['Date'])
            price_data.set_index(['Date'], inplace=True)
        except (OSError, KeyError):
            print('Error loading local price data from:', OPTIONS.price_data)
            sys.exit(-1)
    price_data = price_data.sort_index()
    return price_data

def load_mkt_caps(symbols:list)->Dict:
    mcaps = {} 
    for symbol in symbols:
        print(symbol)
        mcaps[symbol] = yfinance.Ticker(symbol).info['marketCap']
    return mcaps

def main():
    "describe stuff here"
    config = load_config(OPTIONS.config_path)
    prices = load_prices(config)
    mkt_caps = load_mkt_caps(prices.columns.to_list())
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-c', '--config_path', action="store")
    PARSER.add_argument('-d', '--price_data', action="store")
    OPTIONS = PARSER.parse_args()
    main()
