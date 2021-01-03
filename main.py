import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.moment_helpers as mh
import sys
import json
import yfinance
from cvxopt.solvers import qp
from cvxopt import matrix
from joblib import Memory
from pandas_datareader import data as pd_data
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions, CLA
memory = Memory('./cachedir', verbose=0)

def load_config(path):
    "load required config file"
    with open(path) as config_file:
        data = json.load(config_file)
    return data

@memory.cache
def load_prices(symbols, max_lookback_years, data_source, curr_date, config):
    "begin loading prices"
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
            price_data = pd.read_csv(config['price_data'], parse_dates=['Date'])
            price_data.set_index(['Date'], inplace=True)
        except (OSError, KeyError):
            print('Error loading local price data from:', config['price_data'])
            sys.exit(-1)
    price_data = price_data.sort_index()
    return price_data

@memory.cache
def load_mkt_caps(symbols, curr_date):
    print('loading market cap data')
    mcaps = pd_data.get_quote_yahoo(symbols)['marketCap']
    missing_mcap_symbols = mcaps[mcaps.isnull()].index
    for symbol in missing_mcap_symbols:
        print('attempting to find market cap info for', symbol)
        data = yfinance.Ticker(symbol)
        if data.info['quoteType'] == 'ETF' or data.info['quoteType'] == 'MUTUALFUND': 
            mcap = data.info['totalAssets']
            print('adding market cap info for', symbol)
            mcaps.loc[symbol] = mcap
        else:
            print('Failed to find market cap for', symbol)
            sys.exit(-1)
    return mcaps

@memory.cache
def load_market_prices(prices, curr_date):
    mkt_prices = yfinance.download("SPY", period="max")["Adj Close"]
    return mkt_prices
    
def calc_omega(config, symbols):
    variances = []
    for symbol in sorted(symbols):
        view = config['views'][symbol]
        lb, ub  = view[0], view[2]
        std_dev = (ub - lb)/2
        variances.append(std_dev ** 2)
    omega = np.diag(variances)
    return omega

def plot_black_litterman_results(ret_bl, covar_bl, market_prior, mu):
    rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(mu)],
                           index=["Prior", "Posterior", "Views"]).T
    rets_df.plot.bar(figsize=(12,8), title='Black-Litterman Expected Returns');
    plot_heatmap(covar_bl, 'Black-Litterman Covariance', '', '')
    corr_bl = mh.cov2corr(covar_bl)
    corr_bl = pd.DataFrame(corr_bl, index=covar_bl.index, columns=covar_bl.columns)
    plot_heatmap(corr_bl, 'Black-Litterman Correlation', '', '')

def load_mean_views(views, symbols):
    mu = {}
    for symbol in sorted(symbols):
        mu[symbol] = views[symbol][1]
    return mu

def load_data():
    config = load_config(OPTIONS.config_path)
    symbols = sorted(config['views'].keys())
    max_lookback_years = config['max_lookback_years']
    prices = load_prices(symbols, max_lookback_years, config['price_data'], datetime.date.today(), config)
    market_prices = load_market_prices(prices, datetime.date.today())
    mkt_caps = load_mkt_caps(symbols, datetime.date.today())
    return prices, market_prices, mkt_caps, symbols, config

def calc_black_litterman(market_prices, mkt_caps, covar, config, symbols):
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    market_prior = black_litterman.market_implied_prior_returns(mkt_caps, delta, covar)
    mu = load_mean_views(config['views'], symbols)
    omega = calc_omega(config, symbols)
    bl = BlackLittermanModel(covar, pi="market", market_caps=mkt_caps, risk_aversion=delta,
                             absolute_views=mu, omega=omega)
    rets_bl = bl.bl_returns()
    covar_bl = bl.bl_cov()
    plot_black_litterman_results(rets_bl, covar_bl, market_prior, mu);
    return rets_bl, covar_bl

def kelly_optimize(M_df, C_df, config):
    "objective function to maximize is: g(F) = r + F^T(M-R) - F^TCF/2"
    print('Begin Kelly Criterion optimization')
    r = config['annual_risk_free_rate']
    M = M_df.to_numpy()
    C = C_df.to_numpy()

    n = M.shape[0]
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    G = matrix(0.0, (n, n))
    G[::n+1] = -1.0
    h = matrix(0.0, (n, 1))
    try:
        max_pos_size = float(config['max_position_size'])
    except KeyError:
        max_pos_size = None
    try:
        min_pos_size = float(config['min_position_size'])
    except KeyError:
        min_pos_size = None
    if min_pos_size is not None:
        h = matrix(min_pos_size, (n, 1))

    if max_pos_size is not None:
       h_max = matrix(max_pos_size, (n,1))
       G_max = matrix(0.0, (n, n))
       G_max[::n+1] = 1.0
       G = matrix(np.vstack((G, G_max)))
       h = matrix(np.vstack((h, h_max)))

    S = matrix((1.0 / ((1 + r) ** 2)) * C)
    q = matrix((1.0 / (1 + r)) * (M - r))
    sol = qp(S, -q, G, h, A, b)
    kelly = np.array([sol['x'][i] for i in range(n)])
    kelly = pd.DataFrame(kelly, index=C_df.columns, columns=['Weights'])
    kelly = kelly.round(3) 
    kelly.columns=['Kelly']
    return kelly

def max_quad_utility_weights(rets_bl, covar_bl, config):
    print('Begin max quadratic utility optimization')
    returns, sigmas, weights, deltas = [],[],[],[]
    for delta in np.arange(1,10,1):
        ef = EfficientFrontier(rets_bl, covar_bl, weight_bounds= \
                (config['min_position_size'] ,config['max_position_size']))
        ef.max_quadratic_utility(delta)
        ret, sigma, __ = ef.portfolio_performance()
        weights_vec = ef.clean_weights()
        returns.append(ret)
        sigmas.append(sigma)
        deltas.append(delta)
        weights.append(weights_vec)
    fig, ax = plt.subplots()
    ax.plot(sigmas, returns)
    for i, delta in enumerate(deltas):
        ax.annotate(str(delta), (sigmas[i], returns[i]))
    plt.xlabel('Volatility (%) ')
    plt.ylabel('Returns (%)')
    plt.title('Efficient Frontier for Max Quadratic Utility Optimization')
    plt.show()
    opt_delta = float(input('Enter the desired point on the efficient frontier: ') )
    ef = EfficientFrontier(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    ef.max_quadratic_utility(opt_delta)
    opt_weights = ef.clean_weights()
    opt_weights = pd.DataFrame.from_dict(opt_weights, orient='index')
    opt_weights.columns=['Max Quad Util']
    return opt_weights, ef  

def min_volatility_weights(rets_bl, covar_bl, config):
    ef = EfficientFrontier(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    ef.min_volatility()
    weights = ef.clean_weights()
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights.columns=['Min Vol']
    return weights, ef

def max_sharpe_weights(rets_bl, covar_bl, config):
    ef = EfficientFrontier(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    ef.max_sharpe()
    weights = ef.clean_weights()
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights.columns=['Max Sharpe']
    return weights, ef

def cla_max_sharpe_weights(rets_bl, covar_bl, config):
    cla = CLA(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    cla.max_sharpe()
    weights = cla.clean_weights()
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights.columns=['CLA Max Sharpe']
    return weights, cla

def cla_min_vol_weights(rets_bl, covar_bl, config):
    cla = CLA(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    cla.min_volatility()
    weights = cla.clean_weights()
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights.columns=['CLA Min Vol']
    return weights, cla

def plot_heatmap(df, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25, left=0.25)
    heatmap = ax.pcolor(df, edgecolors='w', linewidths=1)
    cbar = plt.colorbar(heatmap)
    ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(df.columns) #, rotation=45)
    ax.set_yticklabels(df.index)

    for y, idx in enumerate(df.index):
        for x, col in enumerate(df.columns):
            plt.text(x + 0.5, y + 0.5, '%.2f' % df.loc[idx, col], \
                     horizontalalignment='center', verticalalignment='center',)

    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def main():
    prices, market_prices, mkt_caps, symbols, config = load_data()

    #covar = risk_models.risk_matrix(prices, method='exp_cov', span=180)
    #covar = risk_models.risk_matrix(prices, method='semicovariance')
    #covar = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    covar = risk_models.risk_matrix(prices, method='oracle_approximating')
    rets_bl, covar_bl = calc_black_litterman(market_prices, mkt_caps, covar, config, symbols)

    kelly_w = kelly_optimize(rets_bl, covar_bl, config) 
    max_quad_util_w, max_quad_util_ef = max_quad_utility_weights(rets_bl, covar_bl, config)
    min_vol_w, min_vol_ef = min_volatility_weights(rets_bl, covar_bl, config)
    max_sharpe_w, max_sharpe_ef = max_sharpe_weights(rets_bl, covar_bl, config)
    cla_max_sharpe_w, cla_max_sharpe_cla = cla_max_sharpe_weights(rets_bl, covar_bl, config)
    cla_min_vol_w, cla_min_vol_cla = cla_min_vol_weights(rets_bl, covar_bl, config)

    #ax = plotting.plot_efficient_frontier(cla_max_sharpe_cla, showfig=False)
    #plt.title('Efficient Frontier via CLA Max Sharpe Optimization')
    #plt.show()
    #ax = plotting.plot_efficient_frontier(cla_min_vol_cla, showfig=False)
    #plt.title('Efficient Frontier via CLA Min Volatility Optimization')
    #plt.show()

    weights_df = pd.merge(kelly_w, max_quad_util_w, left_index=True, right_index=True)
    weights_df = pd.merge(weights_df, max_sharpe_w, left_index=True, right_index=True) 
    weights_df = pd.merge(weights_df, cla_max_sharpe_w, left_index=True, right_index=True) 
    weights_df = pd.merge(weights_df, min_vol_w, left_index=True, right_index=True) 
    weights_df = pd.merge(weights_df, cla_min_vol_w, left_index=True, right_index=True) 
    weights_df.to_csv('portfolio_weight_results.csv')
    
    plot_heatmap(weights_df, 'Portfolio Weighting (%)','Optimization Method', 'Security')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-c', '--config_path', action="store")
    OPTIONS = PARSER.parse_args()
    main()
