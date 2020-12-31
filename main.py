import argparse
import datetime
import numpy as np
import pandas as pd
import sys
import json
import yfinance
from pandas_datareader import data as pd_data
from typing import Dict
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting

def load_config(path):
    "load required config file"
    with open(path) as config_file:
        data = json.load(config_file)
    return data

def load_prices(symbols, max_lookback_years, data_source):
    "load prices from web or from local file"
    if data_source == 'yahoo':
        stock_symbols, crypto_symbols = [], []
        start_date = (datetime.datetime.today()
                      - datetime.timedelta(days=365*max_lookback_years)).date()
        end_date = datetime.datetime.today().date() - datetime.timedelta(days=1)
        symbols = sorted(symbols)
        if len(symbols) > 0:
            print('Downloading adjusted daily close data from Yahoo! Finance')
            try:
                price_data = yfinance.download(symbols, start=str(start_date), end=str(end_date),
                                               interval='1d', auto_adjust=True, threads=True)
            except:
                print('Error downloading data from Yahoo! Finance')
                sys.exit(-1)
            if symbols == ['SPY']:
                cols = [('Close')]
                price_data = price_data[cols]
                price_data.columns = ['SPY']
            else:
                cols = [('Close', x) for x in symbols]
                price_data = price_data[cols]
                price_data.columns = price_data.columns.get_level_values(1)
            price_data.to_csv('sample_data.csv', header=True)
    elif data_source is not None:
        try:
            #Expects a CSV with Date, Symbol header for the prices, i.e. Date, AAPL, GOOGL
            price_data = pd.read_csv(OPTIONS.price_data, parse_dates=['Date'])
            price_data.set_index(['Date'], inplace=True)
        except (OSError, KeyError):
            print('Error loading local price data from:', OPTIONS.price_data)
            sys.exit(-1)
    price_data = price_data.sort_index()
    return price_data

def load_mkt_caps(symbols):
    print('querying market cap data from yahoo')
    mcaps = pd_data.get_quote_yahoo(symbols)['marketCap']
    missing_mcap_symbols = mcaps[mcaps.isnull()].index
    missing_mcaps = pd_data.get_quote_yahoo(missing_mcap_symbols)['NetAssets']
    import pdb
    pdb.set_trace()
    return mcaps

def load_market_prices(prices, max_lookback_years):
    if 'SPY' not in prices.columns:
        return  load_prices(['SPY'], max_lookback_years, OPTIONS.price_data)
    else:
        return prices['SPY'].copy()
    
def calc_omega(config, symbols):
    variances = []
    for symbol in sorted(symbols):
        view = config['views'][symbol]
        lb, ub  = view[0], view[2]
        std_dev = (ub - lb)/2
        variances.append(std_dev ** 2)
    omega = np.diag(variances)
    return omega

def plot_results(rets_df, covar_bl):
    rets_df.plot.bar(figsize=(12,8));
    plotting.plot_covariance(covar_bl);

def load_mean_views(views, symbols):
    mu = {}
    for symbol in sorted(symbols):
        mu[symbol] = views[symbol][1]
    return mu

def main():
    config = load_config(OPTIONS.config_path)
    symbols = sorted(config['views'].keys())
    max_lookback_years = config['max_lookback_years']

    prices = load_prices(symbols, max_lookback_years, OPTIONS.price_data)
    market_prices = load_market_prices(prices, max_lookback_years)
    mkt_caps = load_mkt_caps(symbols)
    #covar = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    #covar = risk_models.risk_matrix(prices, method='exp_cov')
    covar = risk_models.risk_matrix(prices, method='semicovariance')
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    market_prior = black_litterman.market_implied_prior_returns(mkt_caps, delta, covar)
    mu = load_mean_views(config['views'], symbols)
    omega = calc_omega(config, symbols)
    import pdb
    pdb.set_trace()
    bl = BlackLittermanModel(covar, pi="market", market_caps=mkt_caps, risk_aversion=delta,
                             absolute_views=mu, omega=omega)
    ret_bl = bl.bl_returns()
    rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(mu)],
                           index=["Prior", "Posterior", "Views"]).T
    covar_bl = bl.bl_cov()

    plot_results(rets_df, covar_bl)
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-c', '--config_path', action="store")
    PARSER.add_argument('-d', '--price_data', action="store")
    OPTIONS = PARSER.parse_args()
    main()
