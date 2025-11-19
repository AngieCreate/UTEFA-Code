"""
UTEFA QuantiFi - Contestant Template

This template provides the structure for implementing your trading strategy.
Your goal is to maximize portfolio value over 252 (range 0 to 251) trading days.

IMPORTANT:
- Implement your strategy in the update_portfolio() function
- You can store any data you need in the Context class
- Transaction fees apply to both buying and selling (0.5%)
- Do not modify the Market or Portfolio class structures
"""

class Market:
    """
    Represents the stock market with current prices for all available stocks.
    
    Attributes:
        transaction_fee: Float representing the transaction fee (0.5% = 0.005)
        stocks: Dictionary mapping stock names to their current prices
    """
    transaction_fee = 0.005
    
    def __init__(self) -> None:
        # Initialize with 5 stocks
        # Prices will be set by the backtesting script from the CSV data
        self.stocks = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
        }

    def updateMarket(self):
        """
        Updates stock prices to reflect market changes.
        This function will be implemented during grading.
        DO NOT MODIFY THIS METHOD.
        """
        pass


class Portfolio:
    """
    Represents your investment portfolio containing shares and cash.
    
    Attributes:
        shares: Dictionary mapping stock names to number of shares owned
        cash: Float representing available cash balance
    """
    
    def __init__(self) -> None:
        # Start with no shares and $100,000 cash
        self.shares = {
            "Stock_A": 0.0,
            "Stock_B": 0.0,
            "Stock_C": 0.0,
            "Stock_D": 0.0,
            "Stock_E": 0.0
        }
        self.cash = 100000.0

    def evaluate(self, curMarket: Market) -> float:
        """
        Calculate the total value of the portfolio (shares + cash).
        
        Args:
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing total portfolio value
        """
        total_value = self.cash
        
        for stock_name, num_shares in self.shares.items():
            total_value += num_shares * curMarket.stocks[stock_name]
        
        return total_value

    def sell(self, stock_name: str, shares_to_sell: float, curMarket: Market) -> None:
        """
        Sell shares of a specific stock.
        
        Args:
            stock_name: Name of the stock to sell (must match keys in self.shares)
            shares_to_sell: Number of shares to sell (must be positive)
            curMarket: Current Market object with stock prices
            
        Raises:
            ValueError: If shares_to_sell is invalid or exceeds owned shares
        """
        if shares_to_sell <= 0:
            raise ValueError("Number of shares must be positive")

        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")

        if shares_to_sell > self.shares[stock_name]:
            raise ValueError(f"Attempted to sell {shares_to_sell} shares of {stock_name}, but only {self.shares[stock_name]} available")

        # Update portfolio
        self.shares[stock_name] -= shares_to_sell
        sale_proceeds = (1 - Market.transaction_fee) * shares_to_sell * curMarket.stocks[stock_name]
        self.cash += sale_proceeds

    def buy(self, stock_name: str, shares_to_buy: float, curMarket: Market) -> None:
        """
        Buy shares of a specific stock.
        
        Args:
            stock_name: Name of the stock to buy (must match keys in self.shares)
            shares_to_buy: Number of shares to buy (must be positive)
            curMarket: Current Market object with stock prices
            
        Raises:
            ValueError: If shares_to_buy is invalid or exceeds available cash
        """
        if shares_to_buy <= 0:
            raise ValueError("Number of shares must be positive")
        
        if stock_name not in self.shares:
            raise ValueError(f"Invalid stock name: {stock_name}")
        
        cost = (1 + Market.transaction_fee) * shares_to_buy * curMarket.stocks[stock_name]
        
        if cost > self.cash + 0.01:
            raise ValueError(f"Attempted to spend ${cost:.2f}, but only ${self.cash:.2f} available")

        # Update portfolio
        self.shares[stock_name] += shares_to_buy
        self.cash -= cost

    def get_position_value(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to get the current value of a specific position.
        
        Args:
            stock_name: Name of the stock
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing the total value of owned shares for this stock
        """
        return self.shares[stock_name] * curMarket.stocks[stock_name]

    def get_max_buyable_shares(self, stock_name: str, curMarket: Market) -> float:
        """
        Helper method to calculate the maximum number of shares that can be bought.
        
        Args:
            stock_name: Name of the stock
            curMarket: Current Market object with stock prices
            
        Returns:
            Float representing maximum shares that can be purchased with available cash
        """
        price_per_share = curMarket.stocks[stock_name] * (1 + Market.transaction_fee)
        return self.cash / price_per_share if price_per_share > 0 else 0


class Context:
    """
    Store any data you need for your trading strategy.
    
    This class is completely customizable. Use it to track:
    - Historical prices
    - Calculated indicators (moving averages, momentum, etc.)
    - Trading signals
    - Strategy state
    
    Example usage:
        self.price_history = {stock: [] for stock in ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]}
        self.day_counter = 0
    """
    
    def __init__(self) -> None:
        # PUT WHATEVER YOU WANT HERE
        # Example: Track price history for technical analysis
        self.price_history = {
            "Stock_A": [],
            "Stock_B": [],
            "Stock_C": [],
            "Stock_D": [],
            "Stock_E": []
        }
        self.day = 0

        # EMA short and long period amount
        self.short_period = 20
        self.long_period = 100
        # Store EMA history
        self.short_ema = {stock: [] for stock in self.price_history}
        self.long_ema  = {stock: [] for stock in self.price_history}

        # Store trading signals
        self.signals = {stock: [] for stock in self.price_history}

        # Store volatility history
        self.volatility_history = {stock: [] for stock in self.price_history}

        # Store returns history
        self.returns_history = {stock: [] for stock in self.price_history}

def median(values):
    """
    Helper function to calculate the median of a list of numbers.
    
    Args:
        values: List of numerical values
        
    Returns:
        Float representing the median value
    """
    if not values:
         return 0.0
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    else:
        return sorted_vals[mid]

def rolling_std(curMarket: Market, context: Context, period: int = 20):
    """
    Calculate rolling standard deviation of recent returns for each stock.
    append each stock's std to context volatility_history.
    """
    vol_dict = {}
    for stock in curMarket.stocks:
        returns = context.returns_history[stock]
        if len(returns) < period:
            vol_dict[stock] = 0.0
            context.volatility_history[stock].append(0.0)
            continue
        recent_returns = returns[-period:]
        mean_return = sum(recent_returns) / period
        variance = sum((r - mean_return) ** 2 for r in recent_returns) / period
        stddev = variance ** 0.5
        vol_dict[stock] = stddev
        context.volatility_history[stock].append(stddev)
    return vol_dict

def update_portfolio(curMarket: Market, curPortfolio: Portfolio, context: Context):
    """
    Implement your trading strategy here.
    
    This function is called once per trading day, before the market updates.
    
    Args:
        curMarket: Current Market object with stock prices
        curPortfolio: Current Portfolio object with your holdings
        context: Context object for storing strategy data
    
    Example strategy (DO NOT USE THIS - IT'S JUST A PLACEHOLDER):
        # Track prices
        for stock in curMarket.stocks:
            context.price_history[stock].append(curMarket.stocks[stock])
        
        
        # Simple buy-and-hold: invest all cash on day 0
        if context.day == 0:
            for stock in curMarket.stocks:
                max_shares = curPortfolio.get_max_buyable_shares(stock, curMarket)
                if max_shares > 0:
                    curPortfolio.buy(stock, max_shares / 5, curMarket)  # Split equally
        
        context.day += 1
    """
    # YOUR TRADING STRATEGY GOES HERE
    # Appending todays'price to price_history
    for stock in curMarket.stocks:
        context.price_history[stock].append(curMarket.stocks[stock])

    '''
    stock_ranking = {
        "Buy": [],
        "Sell": [],
        "Neutral": [],
        "Buy_Weights": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
        "Sell_Weights": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    }
    '''

    # compute returns and append to returns_history in context class
    for stock in curMarket.stocks:
        prices = context.price_history[stock]
        if len(prices) >= 2:
            prev = prices[-2]
            now = prices[-1]
            if prev == 0:
                r = 0.0
            else:
                r = (now - prev) / prev
            context.returns_history[stock].append(r)
        else:
            # not enough data to compute return; append 0.0
            context.returns_history[stock].append(0.0)
    
    letter_to_stock = { "A": "Stock_A", "B": "Stock_B", "C": "Stock_C", "D": "Stock_D", "E": "Stock_E" }

    # EMA Calculations
    EMA_Calculations(curMarket, context)

    # Volatility Calculations (stddev of recent returns)
    vol_dict = rolling_std(curMarket, context, period=20)
    vol_thresholds = {}
    for stock in curMarket.stocks:
        vol_history = context.volatility_history[stock]
        # take last n = 50 days of volatility history or n = whatever is available for median calculation
        last_n = vol_history[-50:] if len(vol_history) >= 1 else [0.0]

        if len(last_n) < 5:
            vol_thresholds[stock] = float('inf')  # no threshold if insufficient data does not block early trades
        else:
            vol_thresholds[stock] = median(last_n)

    # EMA rankings with volatility filter
    ema_stock_ranking = EMA_Strategy(curMarket, curPortfolio, context)

    # Exceute Trades
    # Calculate weights for buy and sell
    max_buy_percentage = 0.9 # Can sell upto 80% of stocks
    max_sell_percentage = 1.00 # Can sell all stocks
    buy_weights = ema_stock_ranking.get("Buy_Weights", {})
    sell_weights = ema_stock_ranking.get("Sell_Weights", {})
    total_buy_weight = sum(abs(w) for w in buy_weights.values())
    total_sell_weight = sum(abs(w) for w in sell_weights.values())

    # Selling Stocks
    if total_sell_weight > 0:
        for letter, w in sell_weights.items():
            if w <= 0:
                continue
            stock_name = letter_to_stock[letter]
            held_shares = curPortfolio.shares.get(stock_name, 0.0)
            if held_shares <= 0:
                continue
            fraction_to_sell = (w / total_sell_weight) * max_sell_percentage
            vol_factor = 1.0 / (1.0 + vol_dict[stock_name])   # add volatility scaling here
            shares_to_sell = held_shares * fraction_to_sell * vol_factor
            # avoid tiny trades
            if shares_to_sell > 1e-8:
                try:
                    print("day: " + str(context.day) + " | " + str(stock_name) +" sold: " + str(shares_to_sell))
                    curPortfolio.sell(stock_name, shares_to_sell, curMarket)
                    context.signals[stock_name].append(("Sell", context.day))
                except ValueError:
                    # if sell fails due to rounding, skip
                    pass
    
    # Buying Stocks
    if total_buy_weight > 0 and curPortfolio.cash > 0:
        cash_to_spend = curPortfolio.cash * max_buy_percentage
        for letter, w in buy_weights.items():
            if w <= 0:
                continue
            stock_name = letter_to_stock[letter]

            # sclae buy by volatility
            vol = vol_dict[stock_name]
            # only scale if vol > median
            vol_threshold = vol_thresholds[stock]
            if vol > vol_threshold:
                vol_factor = vol_threshold / vol
            else:
                vol_factor = 1.0
            # apply min/max limits
            vol_factor = max(0.3, min(1.0, vol_factor))  # never scale below 30%

            # proportion of cash_to_spend for this stock
            prop = w / total_buy_weight
            cash_alloc = cash_to_spend * prop * vol_factor
            price = curMarket.stocks[stock_name]
            if price <= 0:
                continue
            # convert cash to shares accounting for buy fee
            # cost per share (including fee) = price * (1 + fee)
            cost_per_share = price * (1 + Market.transaction_fee)
            shares_to_buy = cash_alloc / cost_per_share
            # avoid tiny buys
            if shares_to_buy > 1e-8:
                try:
                    print("day: " + str(context.day) + " | " + str(stock_name) + " bought: " + str(shares_to_buy))
                    curPortfolio.buy(stock_name, shares_to_buy, curMarket)
                    context.signals[stock_name].append(("Buy", context.day))
                except ValueError:
                    # not enough cash due to rounding or fee; skip
                    pass

    context.day += 1


def EMA_Calculations(curMarket: Market, context: Context):
    """
    EMA_Calculations calulates the ema for 2 periods, short and long.
    """
    for stock in curMarket.stocks:
        prices = context.price_history[stock]
        price = prices[-1]

        # Calculate Alpha (Smoothing Factor)
        alpha_s = 2 / (context.short_period + 1)
        alpha_l = 2 / (context.long_period + 1)

        # Calculate EMA
        # Short EMA's intial point (Calculated as a simple average)
        if len(prices) == context.short_period:
            init_ema = sum(prices[-context.short_period:]) / context.short_period
            context.short_ema[stock].append(init_ema)
        # Everyday after is calculated Short EMA normally
        elif len(prices) > context.short_period:
            prev = context.short_ema[stock][-1]
            new_ema = alpha_s * price + (1 - alpha_s) * prev
            context.short_ema[stock].append(new_ema)

        # Long EMA's intial point (Calculated as a simple average)
        if len(prices) == context.long_period:
            init_ema = sum(prices[-context.long_period:]) / context.long_period
            context.long_ema[stock].append(init_ema)
        # Everyday after is calculated Long EMA normally
        elif len(prices) > context.long_period:
            prev = context.long_ema[stock][-1]
            new_ema = alpha_l * price + (1 - alpha_l) * prev
            context.long_ema[stock].append(new_ema)

def EMA_Strategy(curMarket: Market, curPortfolio: Portfolio, context: Context):
    """
    EMA_Stragtegy returns a dictionary of a buy list,sell list, neutral (no action) list and 
    weights for buy and sell.
    """
    stocks = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]

    results = {
        "Buy": [],
        "Sell": [],
        "Neutral": [],
        "Buy_Weights": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
        "Sell_Weights": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    }

    # weighing
    bullish_strength = {} # buy strength
    bearish_strength = {} # sell strength

    for stock in curMarket.stocks:
        letter = stock[-1] 
        short_list = context.short_ema[stock]
        long_list  = context.long_ema[stock]

        # Need at least 2 EMA values to detect a crossover
        if len(short_list) < 2 or len(long_list) < 2:
            continue

        s_prev, s_now = short_list[-2], short_list[-1]
        l_prev, l_now = long_list[-2], long_list[-1]

        # Detect crossovers
        bullish_crossover = s_prev < l_prev and s_now > l_now
        bearish_crossover = s_prev > l_prev and s_now < l_now
        neutral = (s_prev > l_prev and s_now > l_now) or (s_prev < l_prev and s_now < l_now)

        # Calculate buy and sell strength
        diff = s_now - l_now
        bull = max(0.0, diff)
        bear = max(0.0, -diff)  # positive when bearish

        # Neutral signal
        if neutral: 
            results["Neutral"].append(letter)

        # Buy signal
        if bullish_crossover: 
            results["Buy"].append(letter)
            bullish_strength[letter] = bull 

        # Sell signal
        if bearish_crossover: 
            results["Sell"].append(letter)
            bearish_strength[letter] = bear 

        # Normalize buy weights independently
        total_bull = sum(bullish_strength.values())
        if total_bull > 0:
            for letter, v in bullish_strength.items():
                results["Buy_Weights"][letter] = v / total_bull
        # Normalize sell weights independently
        total_bear = sum(bearish_strength.values())
        if total_bear > 0:
            for letter, v in bearish_strength.items():
                results["Sell_Weights"][letter] = v / total_bear

    return results

###SIMULATION###
if __name__ == "__main__":
    market = Market()
    portfolio = Portfolio()
    context = Context()

    # Simulate 252 trading days (one trading year)
    for day in range(252):
        update_portfolio(market, portfolio, context)
        market.updateMarket()

    # Print final portfolio value
    final_value = portfolio.evaluate(market)
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(context.signals)