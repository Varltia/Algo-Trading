# Project Introductions

Chen-Yao Hsu

During summer break at the end of 2024, I went back to my home country of Taiwan and to visit family and friends. In the meantime, I was fortunate enough to secure an internship at Fortuna Intelligence Co., LTD as a Quantitative analysis intern. The basic objective of my work was to promote the company’s custom WebSocket and trading API in the form of writing basic trading algorithms and performing backtests on chosen Taiwanese assets via python. Below, I have provided some tailored code snippets of my projects as I believe they provide a modest representation of my skills in coding and mathematics applied in finance.

## Backtest Framework

Realizing that there were various downsides in using backtest packages made for U.S. stock markets, and also seeing a chance to hone my coding skills and my understanding of trading, I decided to create my own backtesting framework tailored to the Taiwanese Stock Exchange. I have pasted its entirety below.


```python
"""
Disillusion v1.0
2024-12-18 FJ Hsu 徐晨耀 erichsu5168@gmail.com

*** WARNING: This framework is only for testing purposes only, and I do not claim that it is by any means representative of live-trading. I am not liable for any inaccuracies and misleading results generated. USE AT YOUR OWN RISK. ***

This is a backtest framework written to simulate stock trading on TWSE, transaction fees and tax are tailored to be the same as trading on the Taiwanese stock market. The framework currently supports long (buying and selling) and short (shorting and buying back the short) positions, and all orders are market orders.

Data
Historical data should be prepared beforehand by the user. Currently, the framework expects a pandas.Dataframe indexed by Datetime with one price per time period. The price of one asset should be headed by a ticker.
Run the code pasted below to get an idea of data required.
# dates = [pd.to_datetime(f'2024-01-{date}') for date in range(1,32)]
# data = {'stock1':[i*100 for i in range(1, 32)], 'stock2':[i*100 for i in range(31, 0, -1)]}
# df = pd.DataFrame(data, columns=['stock1', 'stock2'], index=dates)
df can then be fed into the engine.
sidenote: If you want to be able to feed OHLC data into the engine, I suggest storing the data in the engine as a dictionary ('ticker': pd.Dataframe) and making a few tweaks to the code

Framework
The framework consists of two objects: engine and strategy. An engine is first implemented. After data is fed into the engine, a strategy object can be implemented and fed into the engine. Based on the data for every new period, the strategy implements order objects and stores them, the engine then is responsible for the execution of the orders, keeping track of positions and cash left, and raising warnings. Orders stored in the strategy are deleted after the engine has finished executing, and the process is repeated.

Performance presentation
Market Value of the portfolio, and all positions (with start and end time) throughout the backtest are stored. Currently, I have not provided any graphing and ratios in the framework. However, users can easily use their own metrics and build their own forms of presentation from the results of the engine.
"""
from tqdm import tqdm

class Order:
    """
    An order is created and stored in the strategy when the strategy executes a 'buy', 'sell', 'short_in' or 'short_out' order. Only market type orders are supported currently.
    :param ticker: ticker symbol of stock
    :param size: Order size. In the Taiwanese market, the convention to trade by units of 1000 shares. Thus, when you enter a 1, in actuality its 1000 shares. size is always positive, you do not need to enter a negative when selling.
    :return: None
    """
    def __init__(self, ticker, size, side, idx, type='market'):
        self.ticker = ticker
        self.size = size
        self.side = side
        self.idx = idx
        self.type = type

class Position:
    """
    Position objects are created and stored in the engine when the engine processes orders from the strategy. Positions keep track of the time they are executed, the time they are closed out, the status of the order (active or closed), the margin (initial cash given to the exchange when entering a short, 90% of the trade in this case), and the security (Sell amount of the short minus fees and tax) are also stored.
    :param ticker: ticker symbol of stock
    :param size: size of the order
    :param side: 'long' or 'short' position
    :param idx: time entering the position
    :return: None
    """
    def __init__(self, ticker, size, side, idx, margin=0, security=0):
        self.ticker = ticker
        self.size = size
        self.side = side
        self.idx = idx
        self.margin = margin
        self.security = security
        self.active  = True
        self.close_idx = None

    def __repr__(self):
        if self.active:
            status = 'active'
        else:
            status = 'closed'
        return f'{self.side} {self.size} shares of {self.ticker} at {self.idx}||Status: {status}'

class Strategy:
    """
    The strategy processes data fed into the engine based on the current bars and creates and stores orders when signals are triggered.
    """
    def __init__(self):
        self.current_idx = None
        self.data = None
        self.orders = []

    def add_data(self, data):
        """
        Adds data to the strategy.
        :param data: data passed to the strategy from the engine.
        :return: None
        """
        self.data = data

    def buy(self, ticker, size=1):
        """
        Creates a buy order and stores it in self.orders.
        :param ticker: stock ticker symbol
        :param size: size of the buy order (in thousands)
        :return: None
        """
        self.orders.append(
            Order(
                ticker = ticker,
                side = 'buy',
                size = size,
                idx = self.current_idx
            )
        )

    def sell(self, ticker, size=1):
        """
        Creates a sell order and stores it in self.orders.
        :param ticker: stock ticker symbol
        :param size: size of the sell order (in thousands)
        :return: None
        """
        self.orders.append(
            Order(
                ticker = ticker,
                side = 'sell',
                size = size,
                idx = self.current_idx
            )
        )

    def short_in(self, ticker, size=1):
        """
        Creates a short order and stores it in self.orders.
        :param ticker: stock ticker symbol
        :param size: size of the short order (in thousands)
        :return: None
        """
        self.orders.append(
            Order(
                ticker = ticker,
                side = 'short_in',
                size = size,
                idx = self.current_idx
            )
        )

    def short_out(self, ticker, size=1):
        """
        Creates a close short order and stores it in self.orders.
        :param ticker: stock ticker symbol
        :param size: size of the order (in thousands)
        :return: None
        """
        self.orders.append(
            Order(
                ticker = ticker,
                side = 'short_out',
                size = size,
                idx = self.current_idx
            )
        )

    def on_bar(self):
        """
        This method will be overridden by your strategy when you create your own strategy that inherits from Strategy.
        :return: None
        """
        pass

class Engine:
    """
    The engine takes care of all the calculation in the backtest framework. The engine takes orders created in the strategy and processes them into positions. Positions are then stored in self.positions.
    """
    def __init__(self, initial_cash=1000000):
        self.strategy = None
        self.cash = initial_cash
        self.data = None
        self.current_idx = None
        self.positions = []
        ###
        # These variables can be changed according to your broker
        self.TRADE_FEE = 0.001425 # Transaction fee
        self.TRADE_TAX = 0.003 # Tax incurred when selling stock
        self.BORROW_FEE = 0.001 # Fee to pay when borrowing a stock to short
        ###
        self.portfolio_mv = [initial_cash]
        self.max_cash_deployed = 0

    def add_data(self, data):
        """
        Adds data to the engine.
        :param data: A pd.Dataframe that holds stock prices.
        :return: None
        """
        self.data = data

    def add_strategy(self, strategy):
        """
        Adds a strategy to the engine.
        :param strategy: Strategy object.
        :return: None
        """
        self.strategy = strategy

    def get_current_price(self, ticker):
        """
        Returns the current price of the ticker specified.
        :param ticker: stock ticker symbol
        :return: float, the current price of the ticker.
        """
        return self.data.loc[self.current_idx][ticker]

    def _process_buy(self, order):
        """
        Processes buy orders. Updates cash and positions. Gives warning if there is insufficient cash and does not execute order.
        :param order: Order object.
        :return: None
        """
        if order.side == 'buy':
            if self.cash > self.get_current_price(order.ticker)*order.size*1000*(1+self.TRADE_FEE):
                self.cash -= self.get_current_price(order.ticker)*order.size*1000*(1+self.TRADE_FEE)
                self.positions.append(
                    Position(
                        ticker = order.ticker,
                        side = 'long',
                        size = order.size,
                        idx = self.current_idx
                        )
                    )
                print(f'Buy Executed for {order.size} shares of {order.ticker} at {self.current_idx}.')
            else:
                print(f'Buy Not Executed for {order.size} shares of {order.ticker} at {self.current_idx}: Insufficient Cash.')

    def _process_sell(self, sell):
        """
        Processes sell orders. Checks if there are enough open long positions to close out, if not, it prints a warning and does not execute order. Positions are closed out if they do not meet the size of the order and a later position is processed. If position has a larger size than the order, the position will be closed out and another position is opened with an identical starting time and for the remaining size. Positions closed out will have a closing time stored.
        :param sell: Order object.
        :return: None
        """
        if sell.side == 'sell':
            if sell.size <= sum([i.size for i in self.positions if i.side == 'long' and i.active == True and i.ticker == sell.ticker]):
                print(f'Sell Executed for {sell.size} shares of {sell.ticker} at {self.current_idx}.')
                for position in [i for i in self.positions if i.side == 'long' and i.active == True and i.ticker == sell.ticker]:
                    if position.size < sell.size:
                        position.active = False
                        self.cash += position.size*1000*self.get_current_price(sell.ticker)*(1-self.TRADE_FEE-self.TRADE_TAX)
                        sell.size -= position.size
                        position.close_idx = self.current_idx
                    elif position.size > sell.size:
                        position.active = False
                        position.close_idx = self.current_idx
                        self.cash += sell.size*1000*self.get_current_price(sell.ticker)*(1-self.TRADE_FEE-self.TRADE_TAX)
                        self.positions.append(
                            Position(
                                ticker = sell.ticker,
                                size = position.size-sell.size,
                                side = 'long',
                                idx = position.idx
                            )
                        )
                        sell.size = 0
                    else:
                        position.active = False
                        position.close_idx = self.current_idx
                        self.cash += sell.size*1000*self.get_current_price(sell.ticker)*(1-self.TRADE_FEE-self.TRADE_TAX)
                        sell.size = 0
            else:
                print(f'Sell Not Executed for {sell.size} shares of {sell.ticker} at {self.current_idx}: Insufficient Shares')

    def _process_short_in(self, short):
        """
        Processes short orders. Checks if there are sufficient funds to execute order (Cash has to be larger than the initial margin), if not, it prints a warning and does not execute order. Margin and security are stored in the position for calculations incurred when the position is closed out.
        :param short: Order object.
        :return: None
        """
        if short.side == 'short_in':
            margin = short.size*1000*self.get_current_price(short.ticker)*0.9
            if self.cash >= margin:
                security = short.size*1000*self.get_current_price(short.ticker)*(1-self.TRADE_FEE-self.TRADE_TAX-self.BORROW_FEE)
                self.cash -= margin
                self.positions.append(
                    Position(
                        ticker = short.ticker,
                        size = short.size,
                        side = 'short',
                        idx = short.idx,
                        margin = margin,
                        security = security
                    )
                )
                print(f'Short Executed for {short.size} shares of {short.ticker} at {self.current_idx}.')
            else:
                print(f'Short Not Executed for {short.size} shares of {short.ticker} at {self.current_idx}: Insufficient Cash')

    def _process_short_out(self, short):
        """
        Processes short orders. Checks if there are sufficient shares to execute order, if not, it prints a warning and does not execute order. Positions are closed out if they do not meet the size of the order and a later position is processed. If position has a larger size than the order, the position will be closed out and another position is opened with an identical starting time and for the remaining size. Positions closed out will have a closing time stored.
        :param short: Order object.
        :return: None
        """
        if short.side == 'short_out':
            if short.size <= sum([i.size for i in self.positions if i.ticker == short.ticker and i.side == 'short' and i.active == True]):
                print(f'Short Closed for {short.size} shares of {short.ticker} at {self.current_idx}.')
                for position in [i for i in self.positions if i.ticker == short.ticker and i.side == 'short' and i.active == True]:
                    if position.size < short.size:
                        position.active = False
                        position.close_idx = self.current_idx
                        buy = position.size*1000*self.get_current_price(position.ticker)*(1+self.TRADE_FEE)
                        interest = (position.margin+position.security)*self.BORROW_FEE*(self.current_idx-position.idx).days/365
                        self.cash = self.cash + position.security - buy + interest + position.margin
                        short.size -= position.size
                        position.margin = 0
                        position.security = 0
                    elif position.size > short.size:
                        position.active = False
                        position.close_idx = self.current_idx
                        buy = short.size*1000*self.get_current_price(position.ticker)*(1+self.TRADE_FEE)
                        interest = (position.margin+position.security)*short.size/position.size*self.BORROW_FEE*(self.current_idx-position.idx).days/365
                        self.cash = self.cash + position.security*short.size/position.size - buy + interest + position.margin*short.size/position.size
                        self.positions.append(
                            Position(
                                ticker = short.ticker,
                                side = 'short',
                                size = position.size - short.size,
                                idx = position.idx,
                                margin = position.margin*(position.size - short.size)/position.size,
                                security = position.security*(position.size - short.size)/position.size
                            )
                        )
                        position.margin = 0
                        position.security = 0
                        short.size = 0
                    else:
                        position.active = False
                        position.close_idx = self.current_idx
                        buy = short.size*1000*self.get_current_price(position.ticker)*(1+self.TRADE_FEE)
                        interest = (position.margin+position.security)*short.size/position.size*self.BORROW_FEE*(self.current_idx-position.idx).days/365
                        self.cash = self.cash + position.security*short.size/position.size - buy + interest + position.margin*short.size/position.size
                        position.margin = 0
                        position.security = 0
                        short.size = 0
            else:
                print(f'Short Not Closed for {short.size} shares of {short.ticker} at {self.current_idx}: Insufficient Shares')

    def _update_max_cash_deployed(self):
        """
        This is to calculate the maximum cash deployed at any single period throughout the backtest. Values are stored in self.max_cash_deployed.
        :return: None
        """
        if self.max_cash_deployed < self.portfolio_mv[0] - self.cash:
            self.max_cash_deployed = self.portfolio_mv[0] - self.cash

    def _update_portfolio_mv(self):
        """
        Portfolio market value is recorded for each bar for ease of performance presentation and stored in self.portfolio_mv. On the current bar, cash and value of each active position if closed out is added and appended to self.portfolio_mv.
        :return: None
        """
        value = self.cash
        for position in [i for i in self.positions if i.active]:
            if position.side == 'long':
                value += position.size*1000*self.get_current_price(position.ticker)*(1-self.TRADE_FEE-self.TRADE_TAX)
            elif position.side == 'short':
                value += position.margin
                value = value + position.security - position.size*1000*self.get_current_price(position.ticker)*(1+self.TRADE_FEE) + (position.margin+position.security)*self.BORROW_FEE*(self.current_idx-position.idx).days/365
        self.portfolio_mv.append(value)

    def _check_margin_call(self):
        """
        In the TWSE, margin calls on short positions are triggered when (margin+security)/(current value of stock) drops below 130%. This method is called on every bar to check if current prices trigger any margin calls. If margin calls are triggered, a warning message is printed.
        :return: None
        """
        for position in [i for i in self.positions if i.active and i.side == 'short']:
            if (position.margin+position.security)/position.size*1000*self.get_current_price(position.ticker) < 1.3:
                print(f'Warning: Margin Call on {position.ticker}')

    def _fill_orders(self):
        """
        This method is called on every bar, and is what the engine does in sequence to process all the calculations and updates needed for the backtest. The sequence of the methods called below are important and will affect the accuracy of the backtest.
        :return: None
        """
        self._check_margin_call() # Margin calls should be checked on new prices before any new orders from the strategy are processed.
        # Orders are processed in sequence generated.
        for order in self.strategy.orders:
            self._process_buy(order)
            self._process_sell(order)
            self._process_short_in(order)
            self._process_short_out(order)
        self.strategy.orders = [] # Orders in the strategy are cleaned out after they are processed.
        self._update_max_cash_deployed()
        self._update_portfolio_mv()
        self.positions.sort(key = lambda position: position.idx) # Positions are sorted by their initialized date to ensure sell and closing short orders are executed on the oldest active orders first.

    def run(self):
        """
        This method is called when the user has fed the data and strategy to the engine. It is the command that runs the backtest.
        :return: None
        """
        self.strategy.add_data(self.data) # Data in the engine is fed to the strategy
        for idx in tqdm(self.data.index):
            self.current_idx = idx
            self.strategy.current_idx = self.current_idx
            # Run the strategy on the current bar
            self.strategy.on_bar()
            self._fill_orders()
            # If you want your backtest to take positions on the bar after orders have been called, switch the two lines above
```

## Sample Usage

Here I will create two fictional price series to demonstrate the functionality of the framework.


```python
import pandas as pd
dates = [pd.to_datetime(f'2024-01-{date}') for date in range(1,32)]
data = {'0000':[i*100 for i in range(1, 32)], '9999':[i*100 for i in range(31, 0, -1)]}
df = pd.DataFrame(data, columns=['0000', '9999'], index=dates)
print(df)
```

                0000  9999
    2024-01-01   100  3100
    2024-01-02   200  3000
    2024-01-03   300  2900
    2024-01-04   400  2800
    2024-01-05   500  2700
    2024-01-06   600  2600
    2024-01-07   700  2500
    2024-01-08   800  2400
    2024-01-09   900  2300
    2024-01-10  1000  2200
    2024-01-11  1100  2100
    2024-01-12  1200  2000
    2024-01-13  1300  1900
    2024-01-14  1400  1800
    2024-01-15  1500  1700
    2024-01-16  1600  1600
    2024-01-17  1700  1500
    2024-01-18  1800  1400
    2024-01-19  1900  1300
    2024-01-20  2000  1200
    2024-01-21  2100  1100
    2024-01-22  2200  1000
    2024-01-23  2300   900
    2024-01-24  2400   800
    2024-01-25  2500   700
    2024-01-26  2600   600
    2024-01-27  2700   500
    2024-01-28  2800   400
    2024-01-29  2900   300
    2024-01-30  3000   200
    2024-01-31  3100   100


A simple trading strategy is written. The data and strategy are fed into the framework.


```python
class SimpleStrat(Strategy):
    def on_bar(self):
        if self.current_idx == pd.to_datetime('2024-01-01'):
            self.buy('0000', 5)
        if self.current_idx == pd.to_datetime('2024-01-31'):
            self.sell('0000',5)

e = Engine(initial_cash=10000000)
e.add_data(df)
e.add_strategy(SimpleStrat())
e.run()
```

    100%|██████████| 31/31 [00:00<00:00, 1684.70it/s]

    Buy Executed for 5 shares of 0000 at 2024-01-01 00:00:00.
    Sell Executed for 5 shares of 0000 at 2024-01-31 00:00:00.


    



```python
print(e.cash)
print(e.max_cash_deployed)
print(e.positions)
df_result = df
df_result['port'] = e.portfolio_mv[1:]
df_result.plot.line(y='port')
```

    24930700.0
    500712.5
    [long 5 shares of 0000 at 2024-01-01 00:00:00||Status: closed]





    <Axes: >




    
![png](Project%20Introduction_files/Project%20Introduction_6_2.png)
    


## Strategy Demonstration - Mean Reversion with Kalman Filter and Cointegrating Price Series

In this project, my objective was to demonstrate the effect of dynamic hedge ratios resulting from Kalman filters in contrast to fixed hedge ratios from Johansen test on two cointegrating price series - a Gold Future ETF and a Oil Future ETF. Since the pair did not show extremely high significance levels when testing for cointegration, the Kalman filter seemed helpful in creating a mean reverting portfolio.


### Data Collection

Using the Company's API, I obtained the daily closing prices of the two price series over 2019 to 2023.


```python
from fugle_marketdata import RestClient
import pandas as pd

client = RestClient(api_key = api_key.key) # api_key.key 填入你的富果 RestAPI 鑰匙（api_key = '你的鑰匙')
stock = client.stock

def get_data(symbol, year):
    s = stock.historical.candles(**{"symbol": symbol,
                                  "from": f"{year}-01-01",
                                  "to": f"{year}-12-31",
                                  "fields": "close",
                                  'sort': 'asc'})
    f = pd.DataFrame.from_dict(s['data'])
    f['date'] = pd.to_datetime(f['date'])
    f.set_index('date', inplace=True)
    f.rename(columns={'close':f'{symbol}'}, inplace=True)
    return f

s1 = '00635U'
s2 = '00642U'

s1_19 = get_data(s1, 2019)
s2_19 = get_data(s2, 2019)
s1_20 = get_data(s1, 2020)
s2_20 = get_data(s2, 2020)
s1_21 = get_data(s1, 2021)
s2_21 = get_data(s2, 2021)
s1_22 = get_data(s1, 2022)
s2_22 = get_data(s2, 2022)
s1_23 = get_data(s1, 2023)
s2_23 = get_data(s2, 2023)

df1 = pd.concat([s1_19, s1_20, s1_21, s1_22, s1_23])
df2 = pd.concat([s2_19, s2_20, s2_21, s2_22, s2_23])
df = pd.merge(df1, df2, on='date', how='inner')
```

The plotted price series are shown below.


```python
df.plot()
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](Project%20Introduction_files/Project%20Introduction_12_2.png)
    


### Testing for Cointegration and a Fixed Hedge Ratio

The Johansen test is used to obtain an optimal hedge ratio for a mean reverting portfolio. The ADF test is then used to test if it has significance.


```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

results_johansen = coint_johansen(df, 0, 1)
print(results_johansen.trace_stat)
print(results_johansen.trace_stat_crit_vals)
print(results_johansen.max_eig_stat)
print(results_johansen.max_eig_stat_crit_vals)
```

    [11.99777621  2.20965211]
    [[13.4294 15.4943 19.9349]
     [ 2.7055  3.8415  6.6349]]
    [9.7881241  2.20965211]
    [[12.2971 14.2639 18.52  ]
     [ 2.7055  3.8415  6.6349]]


The port


```python
df['port'] = 0.56741343*df[s1] + 0.18202617*df[s2]
df_BB = df.copy()
df.plot.line(y='port')
```




    <Axes: xlabel='date'>




    
![png](Project%20Introduction_files/Project%20Introduction_16_1.png)
    



```python
from statsmodels.tsa.stattools import adfuller
adfuller(df['port'])
```




    (-2.7504610258468536,
     0.06570587770459516,
     8,
     1207,
     {'1%': -3.43577938005948,
      '5%': -2.863937543790164,
      '10%': -2.568046493171221},
     -1011.3774379817146)



Results show that although the two price series have significant cointegration characteristics, the optimal portfolio has noticeable drift.

### Use of the Kalman Filter

I will not go into detail about the Kalman Filter due to the complexity, but know that it is completed here.


```python
import numpy as np

y = np.array(df[s1])
x = list(df[s2])
x = np.array([[i, 1] for i in x])
delta = 0.0001
yhat = np.array([np.nan for _ in range(len(y))])
e = np.array([np.nan for _ in range(len(y))])
Q = np.array([np.nan for _ in range(len(y))])
P = np.array(np.zeros([2, 2]))
beta = np.array([[np.nan for _ in range(len(y))], [np.nan for _ in range(len(y))]])
Vw = delta / (1 - delta) * np.diag(np.ones(2))
Ve = 0.1
beta[:, 0] = 0
for i in range(1, len(y)):
    beta[:, i] = beta[:, i - 1]  # (1)
    R = P + Vw  # (2)

    yhat[i] = x[i, :] @ beta[:, i]  # (3)
    Q[i] = x[i, :] @ R @ x[i, :].transpose() + Ve  # (4)
    e[i] = y[i] - yhat[i]
    K = R @ x[i, :].transpose() / Q[i]
    beta[:, i] = beta[:, i] + K * e[i]  # (5)
    P = R - np.outer(K, x[i, :] @ R)  #(6)

initial_24 = beta[:, -1]
```


```python
df['slope'] = beta[0,:]
df['intercept'] = beta[1,:]
df['control_port'] = df[s1] + df['slope']*df[s2]
```

The portfolio with a dynamic hege ration is shown here, we can see that drift is removed.


```python
df.plot.line(y='control_port')
```




    <Axes: xlabel='date'>




    
![png](Project%20Introduction_files/Project%20Introduction_22_1.png)
    



```python
df.plot.line(y='intercept')
```




    <Axes: xlabel='date'>




    
![png](Project%20Introduction_files/Project%20Introduction_23_1.png)
    



```python
df.plot.line(y='slope')
```




    <Axes: xlabel='date'>




    
![png](Project%20Introduction_files/Project%20Introduction_24_1.png)
    


### Strategy Development

A trading strategy based on the dynamic hedge ratio is developed and backtested here with the backtest framework from above.


```python
from Disillusion_v1_0 import *

df['sqrt_Q'] = np.sqrt(Q)
df['e'] = e
df['Q_up'] = 2*df['sqrt_Q']
df['Q_low'] = -2*df['sqrt_Q']
df.plot.line(y=['Q_up', 'e', 'Q_low'], ylim=[-4, 8])
```

    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)
    /Users/varltia/PycharmProjects/algo_trading/.venv/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:981: UserWarning: This axis already has a converter set and is updating to a potentially incompatible converter
      return ax.plot(*args, **kwds)





    <Axes: xlabel='date'>




    
![png](Project%20Introduction_files/Project%20Introduction_26_2.png)
    



```python
c = 2

class KF(Strategy):
    def __init__(self):
        super().__init__()
        self.active_position = None
        self.position_size = 0


    def on_bar(self):
        if self.data.loc[self.current_idx, 'e'] <= -c*self.data.loc[self.current_idx, 'sqrt_Q'] and not self.active_position:
            self.buy(s1, 1)
            self.buy(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'long'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] >= c*self.data.loc[self.current_idx, 'sqrt_Q'] and not self.active_position:
            self.short_in(s1, 1)
            self.short_in(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'short'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] >= c*self.data.loc[self.current_idx, 'sqrt_Q'] and self.active_position == 'long':
            self.sell(s1, 1)
            self.sell(s2, self.position_size)
            self.short_in(s1, 1)
            self.short_in(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'short'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        elif self.data.loc[self.current_idx, 'e'] <= -c*self.data.loc[self.current_idx, 'sqrt_Q'] and self.active_position == 'short':
            self.short_out(s1, 1)
            self.short_out(s2, self.position_size)
            self.buy(s1, 1)
            self.buy(s2, self.data.loc[self.current_idx, 'slope'])
            self.active_position = 'long'
            self.position_size = self.data.loc[self.current_idx, 'slope']
        else:
            pass

E = Engine(initial_cash=100000)
E.add_data(df)
E.add_strategy(KF())
E.run()
```

      0%|          | 0/1216 [00:00<?, ?it/s]

    Short Executed for 1 shares of 00635U at 2019-01-03 00:00:00.
    Short Executed for 0.23159211129474916 shares of 00642U at 2019-01-03 00:00:00.
    Short Closed for 1 shares of 00635U at 2019-06-26 00:00:00.
    Short Closed for 0.23159211129474916 shares of 00642U at 2019-06-26 00:00:00.
    Buy Executed for 1 shares of 00635U at 2019-06-26 00:00:00.
    Buy Executed for 1.2050730667303693 shares of 00642U at 2019-06-26 00:00:00.
    Sell Executed for 1 shares of 00635U at 2019-07-03 00:00:00.
    Sell Executed for 1.2050730667303693 shares of 00642U at 2019-07-03 00:00:00.
    Short Executed for 1 shares of 00635U at 2019-07-03 00:00:00.
    Short Executed for 1.1993221633339175 shares of 00642U at 2019-07-03 00:00:00.
    Short Closed for 1 shares of 00635U at 2019-09-10 00:00:00.
    Short Closed for 1.1993221633339175 shares of 00642U at 2019-09-10 00:00:00.
    Buy Executed for 1 shares of 00635U at 2019-09-10 00:00:00.
    Buy Executed for 1.2999167842326227 shares of 00642U at 2019-09-10 00:00:00.
    Sell Executed for 1 shares of 00635U at 2019-09-25 00:00:00.
    Sell Executed for 1.2999167842326227 shares of 00642U at 2019-09-25 00:00:00.
    Short Executed for 1 shares of 00635U at 2019-09-25 00:00:00.
    Short Executed for 1.283459570978833 shares of 00642U at 2019-09-25 00:00:00.
    Short Closed for 1 shares of 00635U at 2020-04-30 00:00:00.
    Short Closed for 1.283459570978833 shares of 00642U at 2020-04-30 00:00:00.
    Buy Executed for 1 shares of 00635U at 2020-04-30 00:00:00.
    Buy Executed for 2.7820204156076733 shares of 00642U at 2020-04-30 00:00:00.
    Sell Executed for 1 shares of 00635U at 2020-06-11 00:00:00.
    Sell Executed for 2.7820204156076733 shares of 00642U at 2020-06-11 00:00:00.
    Short Executed for 1 shares of 00635U at 2020-06-11 00:00:00.
    Short Executed for 2.408512317388625 shares of 00642U at 2020-06-11 00:00:00.
    Short Closed for 1 shares of 00635U at 2020-08-12 00:00:00.
    Short Closed for 2.408512317388625 shares of 00642U at 2020-08-12 00:00:00.
    Buy Executed for 1 shares of 00635U at 2020-08-12 00:00:00.
    Buy Executed for 3.018565486554029 shares of 00642U at 2020-08-12 00:00:00.
    Sell Executed for 1 shares of 00635U at 2020-09-01 00:00:00.
    Sell Executed for 3.018565486554029 shares of 00642U at 2020-09-01 00:00:00.
    Short Executed for 1 shares of 00635U at 2020-09-01 00:00:00.
    Short Executed for 3.039884729934385 shares of 00642U at 2020-09-01 00:00:00.
    Short Closed for 1 shares of 00635U at 2020-11-10 00:00:00.
    Short Closed for 3.039884729934385 shares of 00642U at 2020-11-10 00:00:00.
    Buy Executed for 1 shares of 00635U at 2020-11-10 00:00:00.
    Buy Executed for 3.3185539620135622 shares of 00642U at 2020-11-10 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-01-05 00:00:00.
    Sell Executed for 3.3185539620135622 shares of 00642U at 2021-01-05 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-01-05 00:00:00.
    Short Executed for 2.836194568847844 shares of 00642U at 2021-01-05 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-01-07 00:00:00.
    Short Closed for 2.836194568847844 shares of 00642U at 2021-01-07 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-01-07 00:00:00.
    Buy Executed for 2.796663398548491 shares of 00642U at 2021-01-07 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-03-19 00:00:00.
    Sell Executed for 2.796663398548491 shares of 00642U at 2021-03-19 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-03-19 00:00:00.
    Short Executed for 1.8851062652187287 shares of 00642U at 2021-03-19 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-03-30 00:00:00.
    Short Closed for 1.8851062652187287 shares of 00642U at 2021-03-30 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-03-30 00:00:00.
    Buy Executed for 1.9585639212988464 shares of 00642U at 2021-03-30 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-04-22 00:00:00.
    Sell Executed for 1.9585639212988464 shares of 00642U at 2021-04-22 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-04-22 00:00:00.
    Short Executed for 1.9678395910367485 shares of 00642U at 2021-04-22 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-05-05 00:00:00.
    Short Closed for 1.9678395910367485 shares of 00642U at 2021-05-05 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-05-05 00:00:00.
    Buy Executed for 1.9173711541369671 shares of 00642U at 2021-05-05 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-05-20 00:00:00.
    Sell Executed for 1.9173711541369671 shares of 00642U at 2021-05-20 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-05-20 00:00:00.
    Short Executed for 1.9867153770064956 shares of 00642U at 2021-05-20 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-06-04 00:00:00.
    Short Closed for 1.9867153770064956 shares of 00642U at 2021-06-04 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-06-04 00:00:00.
    Buy Executed for 1.9283238031267609 shares of 00642U at 2021-06-04 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-07-20 00:00:00.
    Sell Executed for 1.9283238031267609 shares of 00642U at 2021-07-20 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-07-20 00:00:00.
    Short Executed for 1.737721744387592 shares of 00642U at 2021-07-20 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-09-16 00:00:00.
    Short Closed for 1.737721744387592 shares of 00642U at 2021-09-16 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-09-16 00:00:00.
    Buy Executed for 1.6803735807762243 shares of 00642U at 2021-09-16 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-11-18 00:00:00.
    Sell Executed for 1.6803735807762243 shares of 00642U at 2021-11-18 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-11-18 00:00:00.
    Short Executed for 1.518809120294583 shares of 00642U at 2021-11-18 00:00:00.
    Short Closed for 1 shares of 00635U at 2021-11-24 00:00:00.
    Short Closed for 1.518809120294583 shares of 00642U at 2021-11-24 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-11-24 00:00:00.
    Buy Executed for 1.5060457797325215 shares of 00642U at 2021-11-24 00:00:00.
    Sell Executed for 1 shares of 00635U at 2021-11-29 00:00:00.
    Sell Executed for 1.5060457797325215 shares of 00642U at 2021-11-29 00:00:00.
    Short Executed for 1 shares of 00635U at 2021-11-29 00:00:00.
    Short Executed for 1.5314619999080086 shares of 00642U at 2021-11-29 00:00:00.


    100%|██████████| 1216/1216 [00:00<00:00, 9063.67it/s]

    Short Closed for 1 shares of 00635U at 2021-12-28 00:00:00.
    Short Closed for 1.5314619999080086 shares of 00642U at 2021-12-28 00:00:00.
    Buy Executed for 1 shares of 00635U at 2021-12-28 00:00:00.
    Buy Executed for 1.5477537759408466 shares of 00642U at 2021-12-28 00:00:00.
    Sell Executed for 1 shares of 00635U at 2022-03-10 00:00:00.
    Sell Executed for 1.5477537759408466 shares of 00642U at 2022-03-10 00:00:00.
    Short Executed for 1 shares of 00635U at 2022-03-10 00:00:00.
    Short Executed for 1.0380309133772214 shares of 00642U at 2022-03-10 00:00:00.
    Short Closed for 1 shares of 00635U at 2022-03-18 00:00:00.
    Short Closed for 1.0380309133772214 shares of 00642U at 2022-03-18 00:00:00.
    Buy Executed for 1 shares of 00635U at 2022-03-18 00:00:00.
    Buy Executed for 1.1150521019125332 shares of 00642U at 2022-03-18 00:00:00.
    Sell Executed for 1 shares of 00635U at 2022-03-31 00:00:00.
    Sell Executed for 1.1150521019125332 shares of 00642U at 2022-03-31 00:00:00.
    Short Executed for 1 shares of 00635U at 2022-03-31 00:00:00.
    Short Executed for 1.0556432419478956 shares of 00642U at 2022-03-31 00:00:00.
    Short Closed for 1 shares of 00635U at 2022-04-14 00:00:00.
    Short Closed for 1.0556432419478956 shares of 00642U at 2022-04-14 00:00:00.
    Buy Executed for 1 shares of 00635U at 2022-04-14 00:00:00.
    Buy Executed for 1.1056486258888572 shares of 00642U at 2022-04-14 00:00:00.
    Sell Executed for 1 shares of 00635U at 2022-06-20 00:00:00.
    Sell Executed for 1.1056486258888572 shares of 00642U at 2022-06-20 00:00:00.
    Short Executed for 1 shares of 00635U at 2022-06-20 00:00:00.
    Short Executed for 0.8660380101850497 shares of 00642U at 2022-06-20 00:00:00.
    Short Closed for 1 shares of 00635U at 2022-10-11 00:00:00.
    Short Closed for 0.8660380101850497 shares of 00642U at 2022-10-11 00:00:00.
    Buy Executed for 1 shares of 00635U at 2022-10-11 00:00:00.
    Buy Executed for 0.8881700714122712 shares of 00642U at 2022-10-11 00:00:00.
    Sell Executed for 1 shares of 00635U at 2022-11-09 00:00:00.
    Sell Executed for 0.8881700714122712 shares of 00642U at 2022-11-09 00:00:00.
    Short Executed for 1 shares of 00635U at 2022-11-09 00:00:00.
    Short Executed for 0.8511480763167013 shares of 00642U at 2022-11-09 00:00:00.
    Short Closed for 1 shares of 00635U at 2023-03-28 00:00:00.
    Short Closed for 0.8511480763167013 shares of 00642U at 2023-03-28 00:00:00.
    Buy Executed for 1 shares of 00635U at 2023-03-28 00:00:00.
    Buy Executed for 1.249544827074747 shares of 00642U at 2023-03-28 00:00:00.
    Sell Executed for 1 shares of 00635U at 2023-04-27 00:00:00.
    Sell Executed for 1.249544827074747 shares of 00642U at 2023-04-27 00:00:00.
    Short Executed for 1 shares of 00635U at 2023-04-27 00:00:00.
    Short Executed for 1.1562374793112045 shares of 00642U at 2023-04-27 00:00:00.
    Short Closed for 1 shares of 00635U at 2023-06-05 00:00:00.
    Short Closed for 1.1562374793112045 shares of 00642U at 2023-06-05 00:00:00.
    Buy Executed for 1 shares of 00635U at 2023-06-05 00:00:00.
    Buy Executed for 1.1870788441743396 shares of 00642U at 2023-06-05 00:00:00.
    Sell Executed for 1 shares of 00635U at 2023-10-12 00:00:00.
    Sell Executed for 1.1870788441743396 shares of 00642U at 2023-10-12 00:00:00.
    Short Executed for 1 shares of 00635U at 2023-10-12 00:00:00.
    Short Executed for 0.8670231384205948 shares of 00642U at 2023-10-12 00:00:00.


    



```python
results = df.copy()
results['mv'] = E.portfolio_mv[1:]
results.plot.line(y='mv')
```




    <Axes: xlabel='date'>




    
![png](Project%20Introduction_files/Project%20Introduction_28_1.png)
    


Results show that the strategy is profitable (with data-snooping bias) over the five year period. Further in this project, I eliminate data-snooping bias by cross-validation using data from 2024. In addition, I also compare dynamic hedge ratio results to results from a backtest with a fixed hedge ratio. Since the process is does not introduce new techniques and this short introduction is for demonstration purposes only, I will omit the remaining details.

Thank You for Reading

Chen-Yao
