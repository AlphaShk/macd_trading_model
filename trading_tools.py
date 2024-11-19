import pandas as pd
import numpy as np
import yaml
import scipy.optimize as opt
import pandas_datareader as pdr

class Metrics(): 
    # A simple container class to store return, volatility, and Sharpe ratio
    def __init__(self, ret, vol, sr): 
        self.ret = ret 
        self.vol = vol 
        self.sr = sr 

class MACD_Model(): 
    def __init__(self, df=None, config_path='config.yaml'): 
        # Initialize the model with a dataframe and configuration file
        self.df = df

        # Load configuration settings from a YAML file
        with open(config_path) as file: 
            config = yaml.safe_load(file)

        # Initialize key parameters based on config
        self.capital = self.initial_capital = config['initial_capital']
        self.alpha = config['alpha']  # Allocation ratio to investment vs cash
        self.ir = config['interest_rate']  # Constant interest rate if ir_const is true
        self.invested = self.alpha * self.capital
        self.cash = (1 - self.alpha) * self.capital 
        self.is_long = True  # Indicates if the portfolio is currently long
        self.res = None  # Placeholder for simulation results
        self.ir_const = config['ir_const']  # Flag for constant interest rate

    def _calculate_returns(self, params, start, end):
        # Calculate returns for a given set of MACD parameters within a time range
        df_old = self.df.copy()  # Preserve original dataframe state
        self.capital = self.initial_capital 
        self.invested = self.alpha * self.capital
        self.cash = (1 - self.alpha) * self.capital 
        self.is_long = True
        self.res = None
        
        self.set_params(params)  # Update moving average parameters
        
        # Run the simulation and return daily returns
        simulation_results, _ = self.run_simulation(start, end)
        self.df = df_old  # Restore original dataframe
        return simulation_results['Daily Return'].dropna()

    def calculate_metrics(self, metric_type, ret=None):
        # Compute annualized metrics: return, volatility, or Sharpe ratio
        if ret is None: 
            ret = self.res['Daily Return']

        # Calculate total and annualized returns
        years = ret.shape[0] / 252  # Assuming 252 trading days per year
        totret = (ret + 1).prod() - 1
        annret = (1 + totret) ** (1 / years) - 1
        
        # Calculate annualized volatility
        vol = ret.std()
        annvol = vol * np.sqrt(252)

        if metric_type == 'return':
            return -annret  # Negative sign for minimization in optimization
        elif metric_type == 'volatility':
            return annvol
        elif metric_type == 'sharpe':
            # Compute risk-free rate based on configuration
            rf = (self.rf + 1).prod() ** (1 / years) - 1 if not self.ir_const else self.ir 
            return -(annret - rf) / annvol  # Negative Sharpe for optimization

    def tune(self, start, end): 
        # Optimize MACD parameters to maximize Sharpe, return, or minimize volatility
        def optimize_metric(metric, initial_guesses, bounds):
            best_metric = np.inf if metric == 'volatility' else -np.inf
            best_params = None

            for guess in initial_guesses:
                # Ensure valid parameter ordering (longer MA > shorter MA)
                if guess[0] <= guess[1]:
                    print(guess)
                    continue
                
                # Minimize the chosen metric using L-BFGS-B method
                result = opt.minimize(lambda params: self.calculate_metrics(
                    metric, self._calculate_returns(params, start, end)), 
                    guess, bounds=bounds, method='L-BFGS-B')  # Changed from Nelder-Mead for efficiency
                
                # Update best parameters and metric value
                metric_value = result.fun if metric == 'volatility' else -result.fun
                if (metric == 'volatility' and metric_value < best_metric) or (
                    metric != 'volatility' and metric_value > best_metric):
                    best_metric = metric_value
                    best_params = result.x
                    print(metric, best_metric, best_params)
            return best_params, best_metric
        
        # Define parameter bounds and initial guesses
        bounds = [(10, 200), (5, 50)]  # Long MA: [10, 200], Short MA: [5, 50]
        initial_guesses = [[l, s] for l in np.linspace(20, 200, 4) 
                           for s in np.linspace(5, 50, 5) if l > s]
        
        # Optimize metrics
        result_s, max_sharpe = optimize_metric('sharpe', initial_guesses, bounds)
        result_r, max_ret = optimize_metric('return', initial_guesses, bounds)
        result_v, min_vol = optimize_metric('volatility', initial_guesses, bounds)

        # Print results
        print("Optimized parameters for:")
        print("Sharpe Ratio: ", result_s)
        print("Return: ", result_r)
        print("Volatility: ", result_v)

        return result_s, result_r, result_v

    def set_params(self, params): 
        # Calculate moving averages based on input parameters
        ma_l, ma_s = params
        self.df['MA Long'] = self.df['Adj Close'].rolling(window=int(ma_l)).mean()
        self.df['MA Short'] = self.df['Adj Close'].rolling(window=int(ma_s)).mean()

    def run_simulation(self, start_date='1920', end_date='2100', data=None): 
        # Simulate trading strategy and compute results
        if not self.ir_const:
            # Fetch risk-free rates for non-constant interest rate scenario
            rf_rates = np.array((pdr.DataReader('IRLTLT01DEM156N', 'fred', start=start_date, end=end_date) 
                                 / 100)['IRLTLT01DEM156N'])
            self.rf = rf_rates = (1 + rf_rates) ** (1/12) - 1  # Monthly compounding

        # Subset dataframe based on date range
        asset_df = self.df if start_date == '1920' and end_date == '2100' else \
                   self.df[(self.df.index >= start_date) & (self.df.index <= end_date)]

        if data is not None: 
            asset_df = data

        # Initialize shares and record initial state
        self.shares = self.invested / asset_df.iloc[0]['Adj Close']
        p = [{'Capital': self.capital, 'Shares': self.shares, 
              'Cash': self.cash, 'Asset': self.invested}]
            
        day = 0
        for date, row in asset_df.iloc[1:].iterrows(): 
            self.trade(row, day, date)
            p.append({'Capital': self.capital, 'Shares': self.shares, 
                      'Cash': self.cash, 'Asset': self.invested})

        # Record results and calculate metrics
        self.res = pd.DataFrame(p, index=asset_df.index)
        self.res['Daily Return'] = self.res['Capital'].pct_change()
        metrics = Metrics(-self.calculate_metrics('return'), 
                          self.calculate_metrics('volatility'), 
                          -self.calculate_metrics('sharpe'))
        return self.res, metrics

    def trade(self, data, day_ind=None, date=None): 
        # Perform trades based on moving average crossover strategy
        if not self.ir_const and day_ind is not None and date is not None and date.day == 1: 
            self.cash *= (1 + self.rf[day_ind])
            day_ind += 1 

        if self.ir_const:
            self.cash *= (1 + self.ir) ** (1 / 252)  # Daily compounding with constant rate 
        
        self.invested = self.shares * data['Adj Close']
        self.capital = self.cash + self.invested

        if pd.notna(data['MA Long']) and pd.notna(data['MA Short']):
            # Execute buy/sell based on crossover
            if data['MA Long'] > data['MA Short'] and self.is_long:
                self._sell(data['Adj Close'])
            elif data['MA Long'] <= data['MA Short'] and not self.is_long:
                self._buy(data['Adj Close'])

    def _buy(self, price): 
        # Buy shares with allocated capital
        self.is_long = True
        self.cash = (1 - self.alpha) * self.capital
        self.invested = self.alpha * self.capital 
        self.shares = self.invested / price
        
    def _sell(self, price): 
        # Sell shares and update cash
        self.is_long = False
        self.cash = self.alpha * self.capital
        self.invested = (1 - self.alpha) * self.capital 
        self.shares = self.invested / price