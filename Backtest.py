""""
Version 1.0
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
import pandas as pd
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

    def _process_buy(self):
        """
        Processes buy orders. Updates cash and positions. Gives warning if there is insufficient cash and does not execute order.
        :return: None
        """
        for i in self.strategy.orders:
            if i.side == 'buy':
                if self.cash > self.get_current_price(i.ticker)*i.size*1000*(1+self.TRADE_FEE):
                    self.cash -= self.get_current_price(i.ticker)*i.size*1000*(1+self.TRADE_FEE)
                    self.positions.append(
                        Position(
                            ticker = i.ticker,
                            side = 'long',
                            size = i.size,
                            idx = self.current_idx
                        )
                    )
                    print(f'Buy Executed for {i.size} shares of {i.ticker} at {self.current_idx}.')
                else:
                    print(f'Buy Not Executed for {i.size} shares of {i.ticker} at {self.current_idx}: Insufficient Cash.')

    def _process_sell(self):
        """
        Processes sell orders. Checks if there are enough open long positions to close out, if not, it prints a warning and does not execute order. Positions are closed out if they do not meet the size of the order and a later position is processed. If position has a larger size than the order, the position will be closed out and another position is opened with an identical starting time and for the remaining size. Positions closed out will have a closing time stored.
        :return: None
        """
        for sell in [i for i in self.strategy.orders if i.side == 'sell']:
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

    def _process_short_in(self):
        """
        Processes short orders. Checks if there are sufficient funds to execute order (Cash has to be larger than the initial margin), if not, it prints a warning and does not execute order. Margin and security are stored in the position for calculations incurred when the position is closed out.
        :return: None
        """
        for short in [i for i in self.strategy.orders if i.side == 'short_in']:
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

    def _process_short_out(self):
        """
        Processes short orders. Checks if there are sufficient shares to execute order, if not, it prints a warning and does not execute order. Positions are closed out if they do not meet the size of the order and a later position is processed. If position has a larger size than the order, the position will be closed out and another position is opened with an identical starting time and for the remaining size. Positions closed out will have a closing time stored.
        :return: None
        """
        for short in [i for i in self.strategy.orders if i.side == 'short_out']:
            if short.size <= sum([i.size for i in self.positions if i.ticker == short.ticker and i.side == 'short' and i.active == True]):
                print(f'Short Closed for {short.size} shares of {short.ticker} at {self.current_idx}.')
                for position in [i for i in self.positions if i.ticker == short.ticker and i.side == 'short' and i.active == True]:
                    if position.size < short.size:
                        position.active = False
                        position.close_idx = self.current_idx
                        buy = position.size*1000*self.get_current_price(position.ticker)*(1+self.TRADE_FEE)
                        interest = (position.margin+position.security)*self.BORROW_FEE*(self.current_idx-position.idx).days/365
                        self.cash = self.cash + position.security - buy + interest + position.deposit
                        short.size -= position.size
                        position.deposit = 0
                        position.security = 0
                    elif position.size > short.size:
                        position.active = False
                        position.close_idx = self.current_idx
                        buy = short.size*1000*self.get_current_price(position.ticker)*(1+self.TRADE_FEE)
                        interest = (position.margin+position.security)*short.size/position.size*self.BORROW_FEE*(self.current_idx-position.idx).days/365
                        self.cash = self.cash + position.security*short.size/position.size - buy + interest + position.deposit*short.size/position.size
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
        self._process_buy()
        self._process_sell()
        self._process_short_in()
        self._process_short_out()
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
            print(idx) # Printing the timestamp on each bar can help us find where problems have occurred in the backtest more efficiently.
            # Run the strategy on the current bar
            self.strategy.on_bar()
            self._fill_orders()
            # If you want your backtest to take positions on the bar after orders have been called, switch the two lines above