<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
        
# [Cardiel](https://thk3421-models.github.io/cardiel/)
Thomas Kirschenmann  
thk3421@gmail.com

# Description
This script is a tool for portfolio managers to input their market forecasts using the Black-Litterman (BL) method, and then use the resulting return vector and covariance matrix estimates as input for optimal portfolio allocations under several different portfolio optimization methods.  The [Black-Litterman model](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model) is a mathematically consistent way to combine a portfolio manager's views on future asset return distributions as a Bayesian prior, which combined with historical market data, produces a posterior distribution for asset returns and covariances.  This is particularly useful, because a portfolio manager may have a view or forecast on individual securities, and a model like BL is required to propogate that view to other securities through an updated covariance matrix and expected return vector. 

This tool will query market data for any security supported by Yahoo! Finance and can also be used with proprietary data in a CSV file.  
The BL return vector and covariance matrix serve as inputs to any standard portfolio optimization methodology, such as Markowitz mean-variance optimization under a variety of utility functions.  This tool calculates several portfolio optimization results for simultaneous comparison, which helps guide a portfolio manager's decision to adjust the portfolio allocation.

# Steps for Using This Tool to Produce Portfolio Allocation Weights 
## Step 1
Create a config.json file.  The example file in the repo contains the necessary fields:  max_lookback_years is simply the maximum number of years to query Yahoo! Finance for historical market data.

