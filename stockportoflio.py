import pandas as pd 
import yfinance as yf  

# Stock lists categorized by risk level
low_risk_stocks = ['JNJ', 'PFE', 'MRK', 'KO', 'PG']  # Safer, stable stocks
medium_risk_stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TMO']  # Growth stocks with moderate risk
high_risk_stocks = ['TSLA', 'NVDA', 'AMD', 'NFLX', 'SQ']  # Volatile stocks with high potential return

# User inputs
investment_amount = float(input("Enter the total investment amount in USD: "))
print("\nSelect your risk level:")
print("1. Low Risk")
print("2. Medium Risk")
print("3. High Risk")
risk_choice = input("Enter 1, 2, or 3: ")

# Select stocks based on user risk preference
if risk_choice == '1':
    selected_stocks = low_risk_stocks
    risk_level = 'Low Risk'
    strategy_name = "Dividend and Stability"
    strategy_description = ("This strategy invests in stable, reliable companies that pay dividends. "
                            "The goal is to provide steady returns with minimal risk by holding these stocks long-term.")
elif risk_choice == '2':
    selected_stocks = medium_risk_stocks
    risk_level = 'Medium Risk'
    strategy_name = "Balanced Growth"
    strategy_description = ("This strategy focuses on growth by investing in well-established companies. "
                            "It aims to balance risk and return, holding stocks for moderate to long-term gains.")
elif risk_choice == '3':
    selected_stocks = high_risk_stocks
    risk_level = 'High Risk'
    strategy_name = "Aggressive Growth"
    strategy_description = ("This strategy targets high-growth stocks with the potential for significant returns. "
                            "It involves higher risk and volatility, holding stocks with strong growth potential.")
else:
    print("Invalid choice. Defaulting to Medium Risk.")
    selected_stocks = medium_risk_stocks
    risk_level = 'Medium Risk'
    strategy_name = "Balanced Growth"
    strategy_description = ("This strategy focuses on growth by investing in well-established companies. "
                            "It aims to balance risk and return, holding stocks for moderate to long-term gains.")

# Equal allocation
allocation = investment_amount / len(selected_stocks)

# Fetch current stock prices
data = yf.download(selected_stocks, period='1d', interval='1d')
current_prices = data['Adj Close'].iloc[0]

# Calculate number of shares to purchase
shares = {}
for stock in selected_stocks:
    num_shares = allocation // current_prices[stock]
    shares[stock] = num_shares

# Portfolio
print(f"\nYour {risk_level} Portfolio:")
for stock, num_shares in shares.items():
    print(f"{stock}: {int(num_shares)} shares")

# Strategy description
strategy_description_full = f"""
Trading Strategy: {strategy_name}
- **Description**: {strategy_description}
"""

print(strategy_description_full)
