# Cardiel - A portfolio allocation tool based on Black-Litterman
Thomas Kirschenmann  
thk3421@gmail.com

## Description
This script is a tool for portfolio managers to input their market forecasts using the Black-Litterman (BL) method, and then use the resulting return vector and covariance matrix estimates as input for optimal portfolio allocations under several different portfolio optimization methods.  The [Black-Litterman model](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model) is a mathematically consistent way to combine a portfolio manager's views on future asset return distributions as a Bayesian prior, which combined with historical market data, produces a posterior distribution for asset returns and covariances.  This is particularly useful, because a portfolio manager may have a view or forecast on individual securities, and a model like BL is required to propogate that view to other securities through an updated covariance matrix and expected return vector. If you have a view on one security, that implies you have a view on other securities because they are all correlated to varying degrees!

This tool will query market data for any security supported by Yahoo! Finance and can also be used with proprietary data in a CSV file.  
The BL return vector and covariance matrix serve as inputs to any standard portfolio optimization methodology, such as Markowitz mean-variance optimization under a variety of utility functions.  This tool calculates the optimal portfolio allocations using several methodologies and presents them simultaneously for side-by-side comparison, which helps guide a portfolio manager's decision to adjust the portfolio allocation.

## Steps for Using This Tool to Produce Portfolio Allocation Weights 
## Step 1: Pip install the required python libraries:
<pre>
pip install argparse matplotlib numpy pandas statsmodels yfinance cvxopt joblib pypfopt
</pre>

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
The "views" field is where the user can entry their views on individual securities in the form of 3 numbers per security.  The first number is the user-provided lower bound annual return for a 1 standard deviation downward move.  The second number is the user-provided estimated annual return, and the third number is the upper bound for a 1 standard deviation upward move.  For example, if a user believes BABA stock is going to have a one-year return of 10% with a lower range forecast of -10% and upper bound forecast of 30% they can enter: "BABA":[-0.10, 0.10, 0.20]. Similarly, the user should repeat this process and enter their views for each asset of interest into their config.json file. A hypothetical example would be: 
<pre>
"views":{
            "BABA":[-0.10, 0.10, 0.20],
            "NVDA":[-0.10, 0.10, 0.30],
            "DIS":[-0.10, 0.07, 0.15],
            "BA":[-0.05, 0.07, 0.15],
            "XOM":[-0.05, 0.07, 0.15],
            "FB":[-0.05, 0.07, 0.15],
            "GOOG":[-0.05, 0.07, 0.15],
            "BAC":[0.0, 0.10, 0.25] 
        }
</pre>
A strong argument can be made that if a portfolio manager does not hold any view whatsoever on a security, then it does not belong in their portfolio!  Of course, if no view is held, then an uninformed prior can be entered by using a very large range for the bounds and the historical data will simply dominate the posterior for that security.

## Step 3: Run the script and review the Black-Litterman results
From a terminal, simply run: 
<pre>
        python main.py --config config.json
</pre>
and the program will load the daily adjusted close prices and compute the Black-Litterman return vector and covariance matrices.  These will be automatically plotted and displayed for validation purposes.  An example summary comparing the views, historical returns, and posterior returns looks like:
![](/example_images/BL_returns.png)
Similarly, the Black-Litterman model covariance and correlations matrices are provided:
![](/example_images/BL_Cov.png)
![](/example_images/BL_corr.png)

### Cached Results ###
One may wish to repeat the above process several times if they are unhappy with the resulting BL returns and covariances.  The Yahoo! query results are automatically cached in local directory, /cachedir, after being received on a given day.  If the program is re-run on the same day with the same set of assets, then cached result will be used.  This will speed up the program by skipping the Yahoo! query and allow for quicker iteration and data exploration.

## Step 4:  Choose a level of risk-aversion
Several optimization routines are automatically run (see details in the portfolio optimization methodology section of this document).  **Most of them require no further input from the user**, however the most commonly used optimization is a Markowitz mean-variance optimization  that requires the user to choose a level of risk-aversion.   This is handled by viewing the [efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier), which is the expected return for an optimal portfolio for a given amount of risk (volatility). The risk aversion parameter is varied, which creates the full curve.  **The user is required to choose a point on the efficient frontier and enter the corresponding number into the terminal.**  
For example:
![](/example_images/EF_max_quad_util.png)
The user seeing the efficient frontier in the chart above may think "I'm okay with 24% volatility for a 10% expected return, so I choose point number 2." Now enter that number into the terminal:

![](/example_images/choose_pt.png)

## Step 5: Compare the portfolio allocations
The tool reports the portfolio allocation weight for each security, using several different optimization schemes.  The portfolio optimizations are all foreced to obey the constraints specified in the config file.  The portfolio optimizations are:
<ul>
        <li> Kelly Criterion: [Kelly objective function](https://en.wikipedia.org/wiki/Kelly_criterion).  Full details of my implementation are discussed here: [thk3421-models.github.io/KellyPortfolio/](https://thk3421-models.github.io/KellyPortfolio/) </li>
        <li> Markowitz Mean-Variance with Maximum Quadratic Utility:  [Markowitz Model](https://en.wikipedia.org/wiki/Markowitz_model#Choosing_the_best_portfolio) </li>
        <li> Maximum Sharpe Ratio: [also known as the Tangency portfolio](http://comisef.wikidot.com/tutorial:tangencyportfolio) </li>
        <li> Minimum Volatility: Portfolio that minimizes the total portfolio volatility </li>
        <li> Critical Line Algorithm - Maximum Sharpe Ratio: [Critical Line Method](https://en.wikipedia.org/wiki/Portfolio_optimization#Specific_approaches) </li>
        <li> Critical Line Algorithm - Minimum Volatility: [Critical Line Method](https://en.wikipedia.org/wiki/Portfolio_optimization#Specific_approaches) </li>
</ul>
The various results should be compared using the final chart produced.

![](/example_images/Portfolio_Weights.png)

This chart is the primary result and purpose of this tool: comparison of optimal portfolio allocations using a variety of methods, all based on the user-provided views that are married to historical data in a Bayesian way through the Black-Litterman model.  The weights are automatically saved to a local CSV file as well.  The user can now look for patterns in the assets and determine if they prefer to increase or decrease a particular holding, whether to reduce expected volatility or to increase potential returns.  

Hope someone finds this useful or interesting! Please send me a note if you want me to add more output or a custom feature! 

[Best wishes, warmest regards!](https://en.wikipedia.org/wiki/Schitt%27s_Creek)


