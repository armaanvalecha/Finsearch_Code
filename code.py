import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

opt = pd.read_csv('OPTIDX_NIFTY_CE_15-May-2025_TO_15-Aug-2025.csv') #option contract data

opt.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True)
opt['date'] = pd.to_datetime(opt['date'], format='%d-%b-%Y')
opt['expiry'] = pd.to_datetime(opt['expiry'], format='%d-%b-%Y')

opt['option_type'] = opt['option_type'].str.upper().map({'CE':'call', 'PE':'put'})

opt['actual_price'] = opt['close']

idx = pd.read_csv('NIFTY 50-15-05-2025-to-15-08-2025.csv') #index data
idx.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True)
idx['date'] = pd.to_datetime(idx['date'], format='%d-%b-%Y')

data = pd.merge(opt, idx[['date', 'close']], on='date', how='left', suffixes=('', '_index'))

data['underlying_price'] = data['close_index']

r = 0.07

data['iv'] = 0.18

data['T'] = (data['expiry'] - data['date']).dt.days / 365

calculated_prices = []
for idx_row, row in data.iterrows():
    S = row['underlying_price']
    K = row['strike_price']
    T = row['T']
    sigma = row['iv']
    option_type = row['option_type']
    price = black_scholes_price(S, K, T, r, sigma, option_type) if pd.notnull(S) else np.nan
    calculated_prices.append(price)

data['bs_price'] = calculated_prices
data['abs_error'] = np.abs(data['bs_price'] - data['actual_price'])

print("Mean Absolute Error:", data['abs_error'].mean())
print(data[['date', 'strike_price', 'option_type', 'actual_price', 'bs_price', 'abs_error']].head(10))

data.to_csv('nifty50_options_bs_comparison.csv', index=False)

sample = data.sample(n=500, random_state=1) if len(data) > 500 else data

plt.figure(figsize=(8, 8))
plt.scatter(sample['actual_price'], sample['bs_price'], alpha=0.5, label='Option Contracts')
plt.plot([sample['actual_price'].min(), sample['actual_price'].max()],
         [sample['actual_price'].min(), sample['actual_price'].max()],
         color='red', linestyle='--', lw=2, label='Perfect Fit (y=x)')
plt.xlabel('Actual Option Price')
plt.ylabel('Black-Scholes Price')
plt.title('Nifty50 Options: Actual vs. Black-Scholes Predicted Price')
plt.legend()
plt.grid(True)
plt.show()
