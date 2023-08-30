# Anchor 1 - Imports Section #  a
# Anchor 1a: Place the import statements here
from binance.client import Client
import os
from os.path import join, dirname
from dotenv import load_dotenv
import numpy as np
import pandas as pd
# Anchor 1z: End of the Imports section

# Anchor 1.5 - Initialize
# Anchor 1.5a: Initialize constants and global variables
short_window = 50
long_window = 200
trade_percentage = 0.05  # 5% of holdings for each trade
in_position = False
current_balance = 5
# Anchor 1.5z: End of Initialize

# Anchor 1.7 - Utility Functions
# Anchor 1.7a: Define utility functions

def calculate_moving_average(data, window):
    return [sum(data[i:i+window])/window for i in range(len(data) - window + 1)]

# Anchor 1.7z: End of Utility Functions

# Anchor 1.8 - Custom Utility Functions
# Anchor 1.8a: Define custom utility functions for advanced algorithms

# Calculate RSI
def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Anchor 1.8z: End of Custom Utility Functions

# Anchor 2 - API Connection
# Anchor 2a: Insert the code for connecting to the Binance API here

# Security Concerns: Handling Sensitive Information
# This section of code handles sensitive information in the form of API keys.
# Proper handling of API keys is essential to prevent unauthorized access to your account.
# It's important to ensure that API keys are kept secure and not exposed publicly.

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Security Concerns: Environment Variables
# The code loads API keys from environment variables stored in the '.env' file.
# Environment variables should never be hard-coded into the code directly to prevent accidental exposure.
# Ensure that the '.env' file is not included in version control to keep the keys private.

API_KEY = os.environ.get("API_KEY_NAME")
API_SECRET = os.environ.get("API_KEY_SECRET")

# Security Concerns: API Key Management
# The values of 'API_KEY_NAME' and 'API_KEY_SECRET' should be properly managed and stored securely.
# Never expose these values directly in code or share them openly in your repository.
# Keep them private to prevent unauthorized access to your trading account.

client = Client(api_key=API_KEY, api_secret=API_SECRET)

try:
    client.ping()
    print("Successfully connected to Binance API!")
except Exception as e:
    # Security Concerns: Error Handling
    # In case of connection failure, it's important not to reveal detailed error messages to the public.
    # Displaying error messages might inadvertently expose internal information or sensitive data.
    print(f"Failed to connect to Binance API. Error: {e}")
# Anchor 2z: End of the API Connection section


# Anchor 3 - Data Retrieval
# Anchor 3a: Fetch historical data

# Fetch minute-level candlestick data for LTCUSDT
candlesticks = client.get_historical_klines("LTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")

# Extract closing prices from the candlestick data
closing_prices = [float(candle[4]) for candle in candlesticks]

# Anchor 3z: End of Data Retrieval section

# Anchor 3.5 - Data Conversion
# Anchor 3.5a: Convert the closing_prices list to a Pandas DataFrame
df = pd.DataFrame(closing_prices, columns=['Close'])
# Anchor 3.5z: End of Data Conversion section

# Anchor 3.7 - Common Calculations
# Anchor 3.7a: Calculate moving averages for both live trading and backtesting
short_moving_avg = calculate_moving_average(closing_prices, short_window)
long_moving_avg = calculate_moving_average(closing_prices, long_window)
# Anchor 3.7z: End of Common Calculations

# Anchor 3.8 - Initialize Machine Learning DataFrame
# Anchor 3.8a: Initialize ml_df with closing_prices
ml_df = pd.DataFrame(closing_prices, columns=['Close'])
# Anchor 3.8z: End of Initialize Machine Learning DataFrame

"""
# Anchor 4 - Balance Management
# Anchor 4a: Function to update balances
def update_balances():
    ltc_info = client.get_asset_balance(asset='LTC')
    usdt_info = client.get_asset_balance(asset='USDT')
    return float(ltc_info['free']), float(usdt_info['free'])

# Fetch and update balances
ltc_balance, usdt_balance = update_balances()
# Anchor 4z: End of Balance Management section
"""
# Anchor 5 - Advanced Trading Strategies
# Anchor 5a: Implementing advanced trading strategies

# Calculate MACD
def calculate_macd(data, short_window, long_window, signal_window):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# Calculate x and MACD
rsi_window = 14
rsi = calculate_rsi(df['Close'], rsi_window)

macd_short_window = 12
macd_long_window = 26
macd_signal_window = 9
macd, signal_line = calculate_macd(df['Close'], macd_short_window, macd_long_window, macd_signal_window)

# Anchor 5b: Handle Missing Values
# Anchor 5b1: Padding missing values with NaN for Short_MA and Long_MA
ml_df['Short_MA'] = np.concatenate([np.full(len(ml_df) - len(short_moving_avg), np.nan), short_moving_avg])
ml_df['Long_MA'] = np.concatenate([np.full(len(ml_df) - len(long_moving_avg), np.nan), long_moving_avg])

# If you have other features that also need padding, you can add them here in a similar manner.
# For example:
# ml_df['Another_Feature'] = np.concatenate([np.full(another_feature_window - 1, np.nan), another_feature])

# Anchor 5b2: (Optional) Drop rows where any of the columns have NaN values if your ML algorithm can't handle them
# ml_df.dropna(inplace=True)

# Anchor 5b3: (Optional) Or you can fill NaN values with a specific value or method
# ml_df.fillna(method='bfill', inplace=True)  # Backward fill
# ml_df.fillna(method='ffill', inplace=True)  # Forward fill
# ml_df.fillna(0, inplace=True)  # Fill with zeros

# Anchor 5bz: End of Handle Missing Values


# Anchor 5z: End of Advanced Trading Strategies

# Anchor 6 - Trading Logic
# Anchor 6a: Define trading functions and logic

def execute_trade(action, symbol="LTC/USDT"):
    global ltc_balance, usdt_balance  # Update the global variables
    if action == "BUY":
        price = float(client.get_symbol_ticker(symbol=symbol)["price"])
        quantity = (usdt_balance * trade_percentage) / price
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        ltc_balance, usdt_balance = update_balances()  # Update balances after trade
    elif action == "SELL":
        quantity = ltc_balance * trade_percentage
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        ltc_balance, usdt_balance = update_balances()  # Update balances after trade
    return order

# Initialize variables to track buy/sell signals
last_short_ma = short_moving_avg[0]
last_long_ma = long_moving_avg[0]

# Main trading loop
for i in range(1, min(len(short_moving_avg), len(long_moving_avg))):
    if not in_position and short_moving_avg[i] > long_moving_avg[i] and last_short_ma <= last_long_ma:
        print("Buy signal at price:", closing_prices[i + long_window - 1])
        #temp real - execute_trade("BUY")
        in_position = True
    elif in_position and short_moving_avg[i] < long_moving_avg[i] and last_short_ma >= last_long_ma:
        print("Sell signal at price:", closing_prices[i + long_window - 1])
        #temp real - execute_trade("SELL")
        in_position = False

    last_short_ma = short_moving_avg[i]
    last_long_ma = long_moving_avg[i]

# Anchor 6b: RSI and MACD
rsi_overbought = 70
rsi_oversold = 30
macd_signal_diff_buy = 0.2
macd_signal_diff_sell = -0.2

# Main trading loop with RSI and MACD
for i in range(max(rsi_window, macd_long_window), len(df)):
    current_rsi = rsi.iloc[i]
    current_macd = macd.iloc[i]
    current_signal = signal_line.iloc[i]
    macd_diff = current_macd - current_signal

    if not in_position:
        if current_rsi < rsi_oversold and macd_diff > macd_signal_diff_buy:
            print("Buy signal based on RSI and MACD at price:", df['Close'].iloc[i])
            #temp real - execute_trade("BUY")
            in_position = True
    elif in_position:
        if current_rsi > rsi_overbought or macd_diff < macd_signal_diff_sell:
            print("Sell signal based on RSI and MACD at price:", df['Close'].iloc[i])
            #temp real - execute_trade("SELL")
            in_position = False

# Anchor 6z: End of Trading Logic section

# Anchor 7 - Backtesting
# Anchor 7a: Initialize backtesting variables
simulated_usdt_balance = 1000  # Starting with 1000 USDT
simulated_ltc_balance = 0  # Starting with 0 LTC
trade_history = []  # To keep track of trade activities

# Anchor 7b: Backtesting loop
for i in range(1, min(len(short_moving_avg), len(long_moving_avg))):
    action = None
    if not in_position and short_moving_avg[i] > long_moving_avg[i] and last_short_ma <= last_long_ma:
        action = "BUY"
    elif in_position and short_moving_avg[i] < long_moving_avg[i] and last_short_ma >= last_long_ma:
        action = "SELL"

    if action:
        price = closing_prices[i + long_window - 1]
        quantity = (current_balance * trade_percentage) / price if action == "BUY" else current_balance * trade_percentage
        trade_history.append({"action": action, "price": price, "quantity": quantity})
        current_balance += quantity * price if action == "SELL" else -quantity * price
        print(f"{action} executed at price {price} with quantity {quantity}, current balance: {current_balance}")

    last_short_ma = short_moving_avg[i]
    last_long_ma = long_moving_avg[i]


# Anchor 7z: End of Backtesting section

# Anchor 8 - Additional Data Collection
# Anchor 8a: Collect Order Book Data

# Fetch the order book for LTCUSDT
order_book = client.get_order_book(symbol='LTCUSDT')

# Extract bids and asks
bids = order_book['bids']
asks = order_book['asks']

# Anchor 8b: Collect Trading Volume Data

# Fetch 24hr ticker for LTCUSDT
ticker_24hr = client.get_ticker(symbol='LTCUSDT')

# Extract trading volume
trading_volume = float(ticker_24hr['volume'])

# Anchor 8z: End of Additional Data Collection

# Anchor 9 - Machine Learning Models
# Anchor 9a: Data Preparation

# Create a new DataFrame for machine learning
# ml_df = df.copy()  # Comment out or remove this line

# Feature Engineering: Add moving averages, RSI, and MACD as features
# ml_df['Short_MA'] = short_moving_avg  # Comment out or remove this line
# ml_df['Long_MA'] = long_moving_avg  # Comment out or remove this line
ml_df['RSI'] = rsi
ml_df['MACD'] = macd

# Drop NaN values
ml_df = ml_df.dropna()

# Data Normalization (Optional)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# ml_df = scaler.fit_transform(ml_df)

# Anchor 9z: End of Data Preparation