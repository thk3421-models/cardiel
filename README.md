<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
        
# [Cardiel](https://thk3421-models.github.io/cardiel/)
Thomas Kirschenmann  
thk3421@gmail.com

# Description
This script is a tool for portfolio managers to input their market forecasts using the Black-Litterman (BL) method, and then use the resulting return vector and covariance matrix estimates as input for optimal portfolio allocations under several different portfolio optimization methods.  The [Black-Litterman model](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model) is a mathematically consistent way to combine a portfolio manager's views on future asset return distributions as a Bayesian prior, which combined with historical market data, produces a posterior distribution for asset returns and covariances.  This is particularly useful, because a portfolio manager may have a view or forecast on individual securities, and a model like BL is required to propogate that view to other securities through an updated covariance matrix and expected return vector. If you have a view on one security, that implies you have a view on other securities because they are all correlated to varying degrees!

This tool will query market data for any security supported by Yahoo! Finance and can also be used with proprietary data in a CSV file.  
The BL return vector and covariance matrix serve as inputs to any standard portfolio optimization methodology, such as Markowitz mean-variance optimization under a variety of utility functions.  This tool calculates the optimal portfolio allocations using several methodologies and presents them simultaneously for side-by-side comparison, which helps guide a portfolio manager's decision to adjust the portfolio allocation.

# Steps for Using This Tool to Produce Portfolio Allocation Weights 
## Step 1: Pip install the required python libraries:
<ul>
        <li> pip install argparse matplotlib numpy pandas statsmodels yfinance cvxopt joblib pypfopt </li>
</ul>

## Step 2: Adjust the included config.json file to your situation.  
The example file in the repo contains the necessary fields:  
<ul>
        <li> max_lookback_years    -- the maximum number of years to query Yahoo! Finance for historical market data </li>
        <li> annual_risk_free_rate -- the annual risk free rate </li>
        <li> max_position_size     -- the maximum allowed position size in percentage terms </li>
        <li> min_position_size     -- the minimum allowed position size in percentage terms </li>
        <li> price_data            -- set this string to "yahoo" to automatically query Yahoo! Finance or provide a path to a CSV file to use proprietary data </li>
        <li> views                 -- See the note below
</ul>
The "views" field is where the user can entry their views on individual securities in the form of 3 numbers per security.  The first number is the user-provided lower bound annual return for a 1 standard deviation downward move.  The second number is the user-provided estimated annual return, and the third number is the upper bound for a 1 standard deviation upward move.  For example, if a user believes AAPL stock is going to have a one-year return of 10% with a lower range forecast of -10% and upper bound forecast of 30%, then enter: 
<pre>
"views":{
        "AAPL":[-0.10, 0.10, 0.20] 
        }
</pre>
into their config.json file.  Continue likewise to enter views on all securities in the portfolio.  A strong argument can be made that if a portfolio manager does not hold any view whatsoever on a security, then it does not belong in their portfolio!  Of course, if no view is held, then an uninformed prior can be entered by using a very large range for the bounds and the historical data will simply dominate the view for that security.

## Step 3: Run the script!
From a terminal, simply run: 
<pre>
        python main.py --config config.json
</pre>
and the program will load the daily adjusted close prices and compute the Black-Litterman return vector and covariance matrices.  These will be automatically plotted and displayed for validation purposes.  An example summary comparing the views, historical returns, and posterior returns looks like:
![]({{/example_images/BL_returns.png)
Similarly, the Black-Litterman model covariance and correlations matrices are provided:
![](/example_images/BL_Cov.png)
![](/example_images/BL_corr.png)

## Step 4:  Choose a level of risk-aversion
Several optimization routines are automatically run (see details in the portfolio optimization methodology section of this document).  **Most of them require no further input from the user**, however the most commonly used optimization is a Markowitz mean-variance optimization  that requires the user to choose a level of risk-aversion.   This is handled by viewing the [efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier), which is the expected return for an optimal portfolio for a given amount of risk (volatility). The risk aversion parameter is varied, which creates the full curve.  **The user is required to choose a point on the efficient frontier and enter the corresponding number into the terminal.**  
For example:
![](/example_images/EF_max_quad_util.png)
the user seeing the efficient frontier in the chart above may think "I'm okay with 24% volatility for a 10% expected return, so I choose point number 2."
![](example_images/choose_pt.png)

## Step 5: Compare the portfolio allocations
The tool reports the portfolio allocation weight for each security, using several different optimization schemes. Currently the portfolio optimizations used are:
<ul>
        <li> Kelly Criterion: [Kelly objective function](https://en.wikipedia.org/wiki/Kelly_criterion).  Full details of my implementation are discussed here: [https://thk3421-models.github.io/KellyPortfolio/](https://thk3421-models.github.io/KellyPortfolio/) </li>
        <li> Markowitz Mean-Variance with Maximum Quadratic Utility:  $$ \max_w w^T \mu - \frac \delta 2 w^T \Sigma w $$, see [Markowitz Model](https://en.wikipedia.org/wiki/Markowitz_model#Choosing_the_best_portfolio) </li>
</ul>
