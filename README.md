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
## Step 1: Adjust the included config.json file to your situation.  
The example file in the repo contains the necessary fields:  
<ul>
        <li> max_lookback_years    -- the maximum number of years to query Yahoo! Finance for historical market data </li>
        <li> annual_risk_free_rate -- the annual risk free rate </li>
        <li> max_position_size     -- the maximum allowed position size in percentage terms </li>
        <li> min_position_size     -- the minimum allowed position size in percentage terms </li>
        <li> price_data            -- use the string "yahoo" to automatically query Yahoo! Finance or provide a path to a CSV file to use proprietary data </li>
        <li> views                 -- See the note below
</ul>
The "views" field is where the user can entry their views on individual securities in the form of 3 numbers per security.  The first number is the user-provided lower bound annual return for a 1 standard deviation downward move.  The second number is the user-provided estimated annual return, and the third number is the upper bound for a 1 standard deviation upward move.  For example, if a user believes AAPL stock is going to have a one-year return of 10% with a lower range forecast of -10% and upper bound forecast of 30%, then they would enter: 

"AAPL":[-0.10, 0.10, 0.20] 

in their config.json file.  Continue likewise to enter views on all securities in the portfolio.  A strong argument can be made that if a portfolio manager does not hold any view whatsoever on a security, then it does not belong in their portfolio!  Of course, if no view is held, then an uninformed prior can be entered by using a very large range for the bounds and the historical data will simply dominate the view for that security.

## Step 2:   
