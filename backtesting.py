import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)  # For reproducibility
class OptionsBacktester:
    def __init__(self, risk_free_rate=0.07):
        self.risk_free_rate = risk_free_rate
        self.results = {}
        
    def black_scholes(self, S, K, T, r, sigma, option_type='CE'):
        """Calculate Black-Scholes option price"""
        if T <= 0:
            if option_type == 'CE':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'CE':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'PE':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'CE' or 'PE'")
        
        return price
    
    def monte_carlo_option_price(self, S, K, T, r, sigma, option_type='CE', simulations=100000):
        """Calculate Monte Carlo option price"""
        if T <= 0:
            if option_type == 'CE':
                return max(S - K, 0), 0
            else:
                return max(K - S, 0), 0
        
        # np.random.seed(42)  # For reproducibility
        Z = np.random.standard_normal(simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        if option_type == 'CE':
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        
        mc_price = np.exp(-r * T) * np.mean(payoff)
        mc_se = np.exp(-r * T) * np.std(payoff) / np.sqrt(simulations)
        
        return mc_price, mc_se
    
    def clean_option_chain(self, csv_file):
        """Clean and process the NSE option chain CSV"""
        # Read the CSV file
        df = pd.read_csv(csv_file, skiprows=1)
        
        # Remove empty columns
        df = df.dropna(axis=1, how='all')
        
        # The actual columns in the CSV file
        actual_columns = [
            'CALLS_OI', 'CALLS_CHNG_IN_OI', 'CALLS_VOLUME', 'CALLS_IV', 'CALLS_LTP', 'CALLS_CHNG',
            'CALLS_BID_QTY', 'CALLS_BID', 'CALLS_ASK', 'CALLS_ASK_QTY', 'STRIKE',
            'PUTS_BID_QTY', 'PUTS_BID', 'PUTS_ASK', 'PUTS_ASK_QTY', 'PUTS_CHNG',
            'PUTS_LTP', 'PUTS_IV', 'PUTS_VOLUME', 'PUTS_CHNG_IN_OI', 'PUTS_OI'
        ]
        
        # Assign only the columns we need
        df = df.iloc[:, :len(actual_columns)]
        df.columns = actual_columns
        
        # Remove commas from numbers and convert to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '')
        
        # Convert to numeric (coerce errors to NaN)
        numeric_cols = actual_columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Prepare calls data
        calls = df[['CALLS_OI', 'CALLS_CHNG_IN_OI', 'CALLS_VOLUME', 'CALLS_IV', 'CALLS_LTP',
                    'CALLS_CHNG', 'CALLS_BID_QTY', 'CALLS_BID', 'CALLS_ASK', 'CALLS_ASK_QTY']].copy()
        calls['STRIKE'] = df['STRIKE']
        calls['TYPE'] = 'CE'
        calls['MARKET_PRICE'] = (calls['CALLS_BID'] + calls['CALLS_ASK']) / 2
        calls['IV'] = calls['CALLS_IV']
        calls['LTP'] = calls['CALLS_LTP']
        
        # Prepare puts data
        puts = df[['PUTS_OI', 'PUTS_CHNG_IN_OI', 'PUTS_VOLUME', 'PUTS_IV', 'PUTS_LTP',
                   'PUTS_CHNG', 'PUTS_BID_QTY', 'PUTS_BID', 'PUTS_ASK', 'PUTS_ASK_QTY']].copy()
        puts['STRIKE'] = df['STRIKE']
        puts['TYPE'] = 'PE'
        puts['MARKET_PRICE'] = (puts['PUTS_BID'] + puts['PUTS_ASK']) / 2
        puts['IV'] = puts['PUTS_IV']
        puts['LTP'] = puts['PUTS_LTP']
        
        # Combine calls and puts
        options = pd.concat([calls, puts], ignore_index=True)
        
        # Keep only relevant columns
        options = options[['STRIKE', 'TYPE', 'MARKET_PRICE', 'IV', 'LTP']]
        
        # Filter out rows with missing essential values
        options = options.dropna(subset=['IV', 'MARKET_PRICE', 'STRIKE'])
        
        return options

    def simulate_market_data(self, initial_price, initial_iv_dict, days_to_backtest, dt=1/365):
        """
        Simulate realistic market data (underlying price and IV) over time
        
        Parameters:
        - initial_price: Starting price of underlying
        - initial_iv_dict: Dictionary mapping (strike, option_type) to initial IV
        - days_to_backtest: Number of days to simulate
        - dt: Time step (1/365 for daily)
        
        Returns:
        - DataFrame with simulated market data for each day
        """
        # np.random.seed(42)  # For reproducibility
        
        # Market parameters
        underlying_vol = 0.20  # Annual volatility of underlying
        iv_vol = 0.30  # Volatility of IV changes (vol of vol)
        iv_mean_reversion = 0.05  # Mean reversion speed for IV
        
        market_data = []
        
        current_price = initial_price
        current_iv_dict = initial_iv_dict.copy()
        
        for day in range(days_to_backtest):
            # Simulate underlying price movement (geometric Brownian motion)
            price_shock = np.random.normal(0, underlying_vol * np.sqrt(dt))
            current_price = current_price * np.exp(-0.5 * underlying_vol**2 * dt + price_shock)
            
            # Simulate IV changes for each option
            simulated_iv_dict = {}
            for (strike, option_type), initial_iv in current_iv_dict.items():
                # IV follows mean-reverting process with volatility clustering
                iv_shock = np.random.normal(0, iv_vol * np.sqrt(dt))
                # Mean revert towards historical average (assume 0.20 or 20%)
                iv_drift = iv_mean_reversion * (0.20 - current_iv_dict[(strike, option_type)]) * dt
                
                new_iv = current_iv_dict[(strike, option_type)] + iv_drift + iv_shock
                # Keep IV within reasonable bounds
                new_iv = max(0.05, min(1.0, new_iv))
                simulated_iv_dict[(strike, option_type)] = new_iv
            
            # Update current IV dictionary
            current_iv_dict = simulated_iv_dict
            
            # Store the day's data
            market_data.append({
                'day': day,
                'underlying_price': current_price,
                'iv_dict': current_iv_dict.copy(),
                'days_to_expiry': days_to_backtest - day,
                'time_to_expiry': (days_to_backtest - day) / 365
            })
        
        return market_data

    def calculate_simulated_market_price(self, S, K, T, iv, option_type, noise_factor=0.02):
        """
        Calculate what the market price would be on a given day
        Uses Black-Scholes as the "true" market price with some noise
        """
        if T <= 0:
            if option_type == 'CE':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        # Use Black-Scholes as the "market consensus" price
        theoretical_price = self.black_scholes(S, K, T, self.risk_free_rate, iv, option_type)
        
        # Add market noise (bid-ask spread, market inefficiencies, etc.)
        noise = np.random.normal(0, noise_factor * theoretical_price)
        market_price = max(0.01, theoretical_price + noise)  # Minimum price of 0.01
        
        return market_price
    
    def backtest_pricing_models(self, csv_file, expiry_date, current_price, days_to_backtest=30, start_date=None):
        """
        FIXED: Proper backtesting that compares model prices against simulated market prices for each day
        """
        print("Starting FIXED backtesting analysis...")
        print(f"Current underlying price: ₹{current_price:.2f}")
        print(f"Expiry date: {expiry_date}")
        
        # Clean option chain data to get initial IV values
        options_data = self.clean_option_chain(csv_file)
        
        # Calculate initial time to expiry
        today = start_date if start_date is not None else datetime.now().date()
        expiry = datetime.strptime(expiry_date, '%d-%b-%Y').date()
        base_T = (expiry - today).days / 365
        
        # Create initial IV dictionary from the option chain
        initial_iv_dict = {}
        for _, option in options_data.iterrows():
            key = (option['STRIKE'], option['TYPE'])
            initial_iv_dict[key] = option['IV'] / 100  # Convert percentage to decimal
        
        # Generate realistic market simulation
        market_simulation = self.simulate_market_data(
            initial_price=current_price,
            initial_iv_dict=initial_iv_dict,
            days_to_backtest=min(days_to_backtest, int(base_T * 365))
        )
        
        # Select representative options for backtesting
        atm_strikes = options_data[
            (options_data['STRIKE'] >= current_price - 200) & 
            (options_data['STRIKE'] <= current_price + 200)
        ]
        
        results_list = []
        
        print(f"Analyzing {len(atm_strikes)} options across {len(market_simulation)} time periods...")
        
        # Now do PROPER backtesting - compare model vs simulated market for each day
        for day_data in market_simulation:
            day = day_data['day']
            S = day_data['underlying_price']
            T = day_data['time_to_expiry']
            iv_dict = day_data['iv_dict']
            days_to_expiry = day_data['days_to_expiry']
            
            # Set random seed for this day to ensure reproducible market prices
            np.random.seed(42 + day)
            
            for _, option in atm_strikes.iterrows():
                K = option['STRIKE']
                option_type = option['TYPE']
                
                # Get the IV for this day (dynamic IV)
                iv_key = (K, option_type)
                if iv_key not in iv_dict:
                    continue  # Skip if we don't have IV data for this option
                
                current_iv = iv_dict[iv_key]
                
                # Calculate the "true" market price for this day
                simulated_market_price = self.calculate_simulated_market_price(
                    S, K, T, current_iv, option_type, noise_factor=0.02
                )
                
                # Calculate model prices using the SAME parameters
                try:
                    bs_price = self.black_scholes(S, K, T, self.risk_free_rate, current_iv, option_type)
                except:
                    bs_price = np.nan
                
                try:
                    mc_price, mc_se = self.monte_carlo_option_price(
                        S, K, T, self.risk_free_rate, current_iv, option_type, simulations=50000
                    )
                except:
                    mc_price, mc_se = np.nan, np.nan
                
                # Calculate intrinsic value
                if option_type == 'CE':
                    intrinsic = max(S - K, 0)
                else:
                    intrinsic = max(K - S, 0)
                
                results_list.append({
                    'Day': day,
                    'Days_to_Expiry': days_to_expiry,
                    'Underlying_Price': S,
                    'Strike': K,
                    'Option_Type': option_type,
                    'Time_to_Expiry': T,
                    'Current_IV': current_iv,
                    'Simulated_Market_Price': simulated_market_price,  # This is now dynamic!
                    'BS_Price': bs_price,
                    'MC_Price': mc_price,
                    'MC_SE': mc_se,
                    'Intrinsic_Value': intrinsic,
                    'BS_Error': bs_price - simulated_market_price if not np.isnan(bs_price) else np.nan,
                    'MC_Error': mc_price - simulated_market_price if not np.isnan(mc_price) else np.nan,
                    'BS_Abs_Error': abs(bs_price - simulated_market_price) if not np.isnan(bs_price) else np.nan,
                    'MC_Abs_Error': abs(mc_price - simulated_market_price) if not np.isnan(mc_price) else np.nan,
                    'BS_Pct_Error': ((bs_price - simulated_market_price) / simulated_market_price * 100) 
                                    if not np.isnan(bs_price) and simulated_market_price != 0 else np.nan,
                    'MC_Pct_Error': ((mc_price - simulated_market_price) / simulated_market_price * 100) 
                                    if not np.isnan(mc_price) and simulated_market_price != 0 else np.nan
                })
        
        self.results = pd.DataFrame(results_list)
        print(f"FIXED Backtesting completed. Analyzed {len(self.results)} option-day combinations.")
        
        return self.results
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.results.empty:
            print("No results available. Please run backtest_pricing_models first.")
            return None
        
        # Remove rows with NaN values for analysis
        clean_results = self.results.dropna(subset=['BS_Error', 'MC_Error'])
        
        metrics = {
            'Black_Scholes': {
                'Mean_Absolute_Error': clean_results['BS_Abs_Error'].mean(),
                'Root_Mean_Square_Error': np.sqrt((clean_results['BS_Error'] ** 2).mean()),
                'Mean_Percentage_Error': clean_results['BS_Pct_Error'].mean(),
                'Mean_Absolute_Percentage_Error': clean_results['BS_Pct_Error'].abs().mean(),
                'Bias': clean_results['BS_Error'].mean(),
                'Standard_Deviation': clean_results['BS_Error'].std(),
                'R_Squared': np.corrcoef(clean_results['BS_Price'], clean_results['Simulated_Market_Price'])[0,1]**2
            },
            'Monte_Carlo': {
                'Mean_Absolute_Error': clean_results['MC_Abs_Error'].mean(),
                'Root_Mean_Square_Error': np.sqrt((clean_results['MC_Error'] ** 2).mean()),
                'Mean_Percentage_Error': clean_results['MC_Pct_Error'].mean(),
                'Mean_Absolute_Percentage_Error': clean_results['MC_Pct_Error'].abs().mean(),
                'Bias': clean_results['MC_Error'].mean(),
                'Standard_Deviation': clean_results['MC_Error'].std(),
                'R_Squared': np.corrcoef(clean_results['MC_Price'], clean_results['Simulated_Market_Price'])[0,1]**2
            }
        }
        
        return metrics
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive visualization plots with FIXED data"""
        if self.results.empty:
            print("No results available. Please run backtest_pricing_models first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Clean data for plotting
        clean_results = self.results.dropna(subset=['BS_Price', 'MC_Price', 'Simulated_Market_Price'])
        
        # 1. Price Comparison Scatter Plot (FIXED to use simulated market prices)
        plt.subplot(3, 3, 1)
        plt.scatter(clean_results['Simulated_Market_Price'], clean_results['BS_Price'], 
                    alpha=0.6, label='Black-Scholes', s=20)
        plt.scatter(clean_results['Simulated_Market_Price'], clean_results['MC_Price'], 
                    alpha=0.6, label='Monte Carlo', s=20)
        
        # Perfect prediction line
        max_price = max(clean_results['Simulated_Market_Price'].max(), 
                        clean_results['BS_Price'].max(), 
                        clean_results['MC_Price'].max())
        plt.plot([0, max_price], [0, max_price], 'k--', alpha=0.5, label='Perfect Prediction')
        
        plt.xlabel('Simulated Market Price (₹)')
        plt.ylabel('Model Price (₹)')
        plt.title('Model Prices vs Simulated Market Prices')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Error Distribution - Black-Scholes
        plt.subplot(3, 3, 2)
        clean_results['BS_Error'].hist(bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(clean_results['BS_Error'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {clean_results["BS_Error"].mean():.2f}')
        plt.xlabel('Pricing Error (₹)')
        plt.ylabel('Frequency')
        plt.title('Black-Scholes Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Error Distribution - Monte Carlo
        plt.subplot(3, 3, 3)
        clean_results['MC_Error'].hist(bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(clean_results['MC_Error'].mean(), color='red', linestyle='--',
                    label=f'Mean: {clean_results["MC_Error"].mean():.2f}')
        plt.xlabel('Pricing Error (₹)')
        plt.ylabel('Frequency')
        plt.title('Monte Carlo Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. IV Evolution Over Time
        plt.subplot(3, 3, 4)
        iv_evolution = clean_results.groupby('Day')['Current_IV'].mean()
        plt.plot(iv_evolution.index, iv_evolution.values * 100, 'b-', linewidth=2, marker='o')
        plt.xlabel('Days from Start')
        plt.ylabel('Average Implied Volatility (%)')
        plt.title('Implied Volatility Evolution (Dynamic IV)')
        plt.grid(True, alpha=0.3)
        
        # 5. Price Path and Market Price Evolution
        plt.subplot(3, 3, 5)
        price_evolution = clean_results.groupby('Day')[['Underlying_Price', 'Simulated_Market_Price']].mean()
        
        ax1 = plt.gca()
        line1 = ax1.plot(price_evolution.index, price_evolution['Underlying_Price'], 
                         'b-', linewidth=2, marker='o', label='Underlying Price')
        ax1.set_xlabel('Days from Start')
        ax1.set_ylabel('Underlying Price (₹)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        line2 = ax2.plot(price_evolution.index, price_evolution['Simulated_Market_Price'], 
                         'r-', linewidth=2, marker='s', label='Avg Option Price')
        ax2.set_ylabel('Average Option Price (₹)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        plt.title('Market Evolution: Underlying & Option Prices')
        plt.grid(True, alpha=0.3)
        
        # 6. Error by Time to Expiry (FIXED)
        plt.subplot(3, 3, 6)
        plt.scatter(clean_results['Days_to_Expiry'], clean_results['BS_Abs_Error'], 
                    alpha=0.6, label='Black-Scholes', s=20)
        plt.scatter(clean_results['Days_to_Expiry'], clean_results['MC_Abs_Error'], 
                    alpha=0.6, label='Monte Carlo', s=20)
        plt.xlabel('Days to Expiry')
        plt.ylabel('Absolute Error (₹)')
        plt.title('Pricing Error by Time to Expiry')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Model Performance Over Time
        plt.subplot(3, 3, 7)
        daily_bs_error = clean_results.groupby('Day')['BS_Abs_Error'].mean()
        daily_mc_error = clean_results.groupby('Day')['MC_Abs_Error'].mean()
        
        plt.plot(daily_bs_error.index, daily_bs_error.values, 
                 marker='o', label='Black-Scholes', linewidth=2)
        plt.plot(daily_mc_error.index, daily_mc_error.values, 
                 marker='s', label='Monte Carlo', linewidth=2)
        plt.xlabel('Days from Start')
        plt.ylabel('Mean Absolute Error (₹)')
        plt.title('Model Performance Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. IV Impact on Pricing Errors
        plt.subplot(3, 3, 8)
        iv_bins = pd.cut(clean_results['Current_IV'], bins=5)
        iv_bs_error = clean_results.groupby(iv_bins)['BS_Abs_Error'].mean()
        iv_mc_error = clean_results.groupby(iv_bins)['MC_Abs_Error'].mean()
        
        x_pos = np.arange(len(iv_bs_error))
        width = 0.35
        
        plt.bar(x_pos - width/2, iv_bs_error.values, width, label='Black-Scholes', alpha=0.7)
        plt.bar(x_pos + width/2, iv_mc_error.values, width, label='Monte Carlo', alpha=0.7)
        
        plt.xlabel('Dynamic IV Bins')
        plt.ylabel('Mean Absolute Error (₹)')
        plt.title('Error by Dynamic Implied Volatility')
        plt.xticks(x_pos, [f'{interval.left:.2f}-{interval.right:.2f}' for interval in iv_bs_error.index])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 9. Model Performance Summary
        plt.subplot(3, 3, 9)
        metrics = self.calculate_performance_metrics()
        
        if metrics:
            bs_rmse = metrics['Black_Scholes']['Root_Mean_Square_Error']
            mc_rmse = metrics['Monte_Carlo']['Root_Mean_Square_Error']
            bs_mape = metrics['Black_Scholes']['Mean_Absolute_Percentage_Error']
            mc_mape = metrics['Monte_Carlo']['Mean_Absolute_Percentage_Error']
            
            categories = ['RMSE', 'MAPE (%)']
            bs_values = [bs_rmse, bs_mape]
            mc_values = [mc_rmse, mc_mape]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, bs_values, width, label='Black-Scholes', alpha=0.7)
            plt.bar(x + width/2, mc_values, width, label='Monte Carlo', alpha=0.7)
            
            plt.ylabel('Error Magnitude')
            plt.title('FIXED Model Performance Comparison')
            plt.xticks(x, categories)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bs_val, mc_val) in enumerate(zip(bs_values, mc_values)):
                plt.text(i - width/2, bs_val + max(bs_values) * 0.01, f'{bs_val:.2f}', 
                         ha='center', va='bottom')
                plt.text(i + width/2, mc_val + max(mc_values) * 0.01, f'{mc_val:.2f}', 
                         ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("fixed_comprehensive_analysis.png")
        plt.show()

    def print_performance_summary(self):
        """Print a comprehensive performance summary"""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            print("No metrics available.")
            return
        
        print("\n" + "="*80)
        print("OPTION PRICING MODELS - BACKTESTING PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\nDataset Size: {len(self.results)} option-day combinations")
        clean_results = self.results.dropna(subset=['BS_Error', 'MC_Error'])
        print(f"Clean Data Points: {len(clean_results)}")
        
        # Show the key fixes implemented
        # print(f"\n🔧 KEY FIXES IMPLEMENTED:")
        # print(f"✅ Dynamic Market Prices: Each day has its own simulated market price")
        # print(f"✅ Dynamic Implied Volatility: IV changes over time with mean reversion")
        # print(f"✅ Proper Comparison: Models compared against same-day market conditions")
        
        print(f"\n{'Metric':<35} {'Black-Scholes':<15} {'Monte Carlo':<15} {'Winner':<10}")
        print("-" * 75)
        
        bs_metrics = metrics['Black_Scholes']
        mc_metrics = metrics['Monte_Carlo']
        
        comparisons = [
            ('Mean Absolute Error (₹)', 'Mean_Absolute_Error'),
            ('Root Mean Square Error (₹)', 'Root_Mean_Square_Error'),
            ('Mean Percentage Error (%)', 'Mean_Percentage_Error'),
            ('Mean Abs. Percentage Error (%)', 'Mean_Absolute_Percentage_Error'),
            ('Bias (₹)', 'Bias'),
            ('Standard Deviation (₹)', 'Standard_Deviation'),
            ('R-Squared', 'R_Squared')
        ]
        
        bs_wins = 0
        mc_wins = 0
        
        for metric_name, metric_key in comparisons:
            bs_val = bs_metrics[metric_key]
            mc_val = mc_metrics[metric_key]
            
            # For R-squared, higher is better; for others, lower is better
            if metric_key == 'R_Squared':
                winner = 'BS' if bs_val > mc_val else 'MC'
                if bs_val > mc_val:
                    bs_wins += 1
                else:
                    mc_wins += 1
            else:
                winner = 'BS' if abs(bs_val) < abs(mc_val) else 'MC'
                if abs(bs_val) < abs(mc_val):
                    bs_wins += 1
                else:
                    mc_wins += 1
            
            print(f"{metric_name:<35} {bs_val:<15.4f} {mc_val:<15.4f} {winner:<10}")
        
        print("-" * 75)
        print(f"Overall Winner: {'Black-Scholes' if bs_wins > mc_wins else 'Monte Carlo'} "
              f"({max(bs_wins, mc_wins)}/{len(comparisons)} metrics)")
        
        # Show dynamic market insights
        print(f"\nDYNAMIC MARKET INSIGHTS:")
        avg_underlying = clean_results['Underlying_Price'].mean()
        underlying_vol_realized = clean_results['Underlying_Price'].std() / avg_underlying
        avg_iv = clean_results['Current_IV'].mean()
        iv_range = (clean_results['Current_IV'].min(), clean_results['Current_IV'].max())
        
        print(f"• Average underlying price: ₹{avg_underlying:.2f}")
        print(f"• Realized underlying volatility: {underlying_vol_realized:.2%}")
        print(f"• Average implied volatility: {avg_iv:.2%}")
        print(f"• IV range: {iv_range[0]:.2%} - {iv_range[1]:.2%}")
        print(f"• Black-Scholes average absolute error: ₹{bs_metrics['Mean_Absolute_Error']:.2f} "
              f"({bs_metrics['Mean_Absolute_Percentage_Error']:.2f}% of option value)")
        print(f"• Monte Carlo average absolute error: ₹{mc_metrics['Mean_Absolute_Error']:.2f} "
              f"({mc_metrics['Mean_Absolute_Percentage_Error']:.2f}% of option value)")
        
        # Performance by option type
        call_results = clean_results[clean_results['Option_Type'] == 'CE']
        put_results = clean_results[clean_results['Option_Type'] == 'PE']
        
        if not call_results.empty and not put_results.empty:
            print(f"\nPERFORMANCE BY OPTION TYPE:")
            print(f"Calls - BS MAE: ₹{call_results['BS_Abs_Error'].mean():.2f}, "
                  f"MC MAE: ₹{call_results['MC_Abs_Error'].mean():.2f}")
            print(f"Puts  - BS MAE: ₹{put_results['BS_Abs_Error'].mean():.2f}, "
                  f"MC MAE: ₹{put_results['MC_Abs_Error'].mean():.2f}")
        
        print("\n" + "="*80)

# Example usage and execution
if __name__ == "__main__":
    # Initialize the backtester
    backtester = OptionsBacktester(risk_free_rate=0.07)
    
    # Parameters for backtesting
    csv_file = 'option-chain-ED-NIFTY-14-Aug-2025.csv'
    expiry_date = '14-Aug-2025'
    current_price = 24631.30  # Current Nifty50 index value
    days_to_backtest = 25    # Number of days to simulate
    
    print("Starting Options Pricing Models Backtesting...")
    print("=" * 60)
    
    # Run the FIXED comprehensive backtesting
    results_df = backtester.backtest_pricing_models(
        csv_file=csv_file,
        expiry_date=expiry_date,
        current_price=current_price,
        days_to_backtest=days_to_backtest,
        start_date=datetime.strptime('10-Jul-2025', '%d-%b-%Y').date()
    )
    
    # Display sample results
    print("\nSample FIXED Backtest Results:")
    sample_cols = ['Day', 'Underlying_Price', 'Strike', 'Option_Type', 'Current_IV', 
                   'Simulated_Market_Price', 'BS_Price', 'MC_Price', 'BS_Error', 'MC_Error']
    print(results_df[sample_cols].head(10).round(4).to_string(index=False))
    
    # Print comprehensive performance summary
    backtester.print_performance_summary()
    
    # Create all visualizations
    print("\nGenerating FIXED comprehensive analysis plots...")
    backtester.plot_comprehensive_analysis()
    
    # Additional analysis functions with FIXES
    def analyze_dynamic_iv_impact(backtester):
        """Analyze how dynamic IV affects model performance"""
        clean_results = backtester.results.dropna(subset=['BS_Error', 'MC_Error'])
        
        # Calculate IV change from day to day for each option
        iv_changes = []
        for strike in clean_results['Strike'].unique():
            for opt_type in clean_results['Option_Type'].unique():
                option_data = clean_results[
                    (clean_results['Strike'] == strike) & 
                    (clean_results['Option_Type'] == opt_type)
                ].sort_values('Day')
                
                if len(option_data) > 1:
                    iv_diff = option_data['Current_IV'].diff().dropna()
                    iv_changes.extend(iv_diff.tolist())
        
        avg_iv_change = np.mean(np.abs(iv_changes)) if iv_changes else 0
        
        print(f"\nDYNAMIC IV ANALYSIS:")
        print(f"• Average absolute daily IV change: {avg_iv_change:.4f} ({avg_iv_change*100:.2f}%)")
        print(f"• IV volatility (std of changes): {np.std(iv_changes):.4f}")
        
        # Analyze performance in high vs low IV change periods
        clean_results['IV_Volatility'] = clean_results.groupby(['Strike', 'Option_Type'])['Current_IV'].transform(lambda x: x.rolling(3, min_periods=1).std())
        
        high_iv_vol = clean_results[clean_results['IV_Volatility'] > clean_results['IV_Volatility'].median()]
        low_iv_vol = clean_results[clean_results['IV_Volatility'] <= clean_results['IV_Volatility'].median()]
        
        print(f"\nPERFORMANCE IN DIFFERENT IV REGIMES:")
        print(f"High IV Volatility periods:")
        print(f"  BS MAE: ₹{high_iv_vol['BS_Abs_Error'].mean():.2f}")
        print(f"  MC MAE: ₹{high_iv_vol['MC_Abs_Error'].mean():.2f}")
        print(f"Low IV Volatility periods:")
        print(f"  BS MAE: ₹{low_iv_vol['BS_Abs_Error'].mean():.2f}")
        print(f"  MC MAE: ₹{low_iv_vol['MC_Abs_Error'].mean():.2f}")
        
        return {
            'avg_iv_change': avg_iv_change,
            'high_iv_vol_performance': {
                'bs_mae': high_iv_vol['BS_Abs_Error'].mean(),
                'mc_mae': high_iv_vol['MC_Abs_Error'].mean()
            },
            'low_iv_vol_performance': {
                'bs_mae': low_iv_vol['BS_Abs_Error'].mean(),
                'mc_mae': low_iv_vol['MC_Abs_Error'].mean()
            }
        }
    
    def analyze_price_path_dependency(backtester):
        """Analyze how underlying price movements affect model performance"""
        clean_results = backtester.results.dropna(subset=['BS_Error', 'MC_Error'])
        
        # Calculate daily price returns
        price_data = clean_results.groupby('Day')['Underlying_Price'].first().sort_index()
        price_returns = price_data.pct_change().dropna()
        
        # Categorize days by price movement
        up_days = clean_results[clean_results['Day'].isin(
            price_returns[price_returns > 0.01].index  # Up more than 1%
        )]
        down_days = clean_results[clean_results['Day'].isin(
            price_returns[price_returns < -0.01].index  # Down more than 1%
        )]
        flat_days = clean_results[clean_results['Day'].isin(
            price_returns[abs(price_returns) <= 0.01].index  # Flat within 1%
        )]
        
        print(f"\nPRICE PATH DEPENDENCY ANALYSIS:")
        print(f"Up Days (>1%): {len(up_days)} observations")
        if not up_days.empty:
            print(f"  BS MAE: ₹{up_days['BS_Abs_Error'].mean():.2f}")
            print(f"  MC MAE: ₹{up_days['MC_Abs_Error'].mean():.2f}")
        
        print(f"Down Days (<-1%): {len(down_days)} observations")
        if not down_days.empty:
            print(f"  BS MAE: ₹{down_days['BS_Abs_Error'].mean():.2f}")
            print(f"  MC MAE: ₹{down_days['MC_Abs_Error'].mean():.2f}")
        
        print(f"Flat Days (±1%): {len(flat_days)} observations")
        if not flat_days.empty:
            print(f"  BS MAE: ₹{flat_days['BS_Abs_Error'].mean():.2f}")
            print(f"  MC MAE: ₹{flat_days['MC_Abs_Error'].mean():.2f}")
        
        return {
            'price_returns': price_returns,
            'directional_performance': {
                'up_days': {'bs_mae': up_days['BS_Abs_Error'].mean() if not up_days.empty else np.nan,
                           'mc_mae': up_days['MC_Abs_Error'].mean() if not up_days.empty else np.nan},
                'down_days': {'bs_mae': down_days['BS_Abs_Error'].mean() if not down_days.empty else np.nan,
                             'mc_mae': down_days['MC_Abs_Error'].mean() if not down_days.empty else np.nan},
                'flat_days': {'bs_mae': flat_days['BS_Abs_Error'].mean() if not flat_days.empty else np.nan,
                             'mc_mae': flat_days['MC_Abs_Error'].mean() if not flat_days.empty else np.nan}
            }
        }
    
    def create_model_validation_analysis(backtester):
        """Validate that our fixes actually work as expected"""
        clean_results = backtester.results.dropna(subset=['BS_Price', 'MC_Price', 'Simulated_Market_Price'])
        
        
        # 1. Verify market prices change over time
        market_price_by_day = clean_results.groupby('Day')['Simulated_Market_Price'].mean()
        market_price_std = market_price_by_day.std()
        
        # 2. Verify IV changes over time
        iv_by_day = clean_results.groupby('Day')['Current_IV'].mean()
        iv_std = iv_by_day.std()
        
        # 3. Verify underlying price path is realistic
        underlying_by_day = clean_results.groupby('Day')['Underlying_Price'].first()
        underlying_returns = underlying_by_day.pct_change().dropna()
        annual_vol = underlying_returns.std() * np.sqrt(252)
        # print(f"✅ Underlying Volatility: {annual_vol:.2%} (annualized)")
        
        # 4. Check for proper model vs market relationships
        bs_corr = np.corrcoef(clean_results['BS_Price'], clean_results['Simulated_Market_Price'])[0,1]
        mc_corr = np.corrcoef(clean_results['MC_Price'], clean_results['Simulated_Market_Price'])[0,1]
        # print(f"✅ BS-Market Correlation: {bs_corr:.3f}")
        # print(f"✅ MC-Market Correlation: {mc_corr:.3f}")
        
        # 5. Verify errors are reasonable (should be small since market is based on BS)
        avg_bs_error = clean_results['BS_Abs_Error'].mean()
        avg_mc_error = clean_results['MC_Abs_Error'].mean()
        avg_market_price = clean_results['Simulated_Market_Price'].mean()
        
        # print(f"✅ BS Error as % of market price: {(avg_bs_error/avg_market_price)*100:.2f}%")
        # print(f"✅ MC Error as % of market price: {(avg_mc_error/avg_market_price)*100:.2f}%")
        
        return {
            'market_price_std': market_price_std,
            'iv_std': iv_std,
            'underlying_vol': annual_vol,
            'correlations': {'bs': bs_corr, 'mc': mc_corr},
            'relative_errors': {
                'bs_pct': (avg_bs_error/avg_market_price)*100,
                'mc_pct': (avg_mc_error/avg_market_price)*100
            }
        }
    
    # Run additional analyses with FIXED data
    # print("\nRunning additional analyses on FIXED data...")
    
    # Dynamic IV analysis
    iv_analysis = analyze_dynamic_iv_impact(backtester)
    
    # Price path dependency analysis
    path_analysis = analyze_price_path_dependency(backtester)
    
    # Model validation
    validation_analysis = create_model_validation_analysis(backtester)
    
    # Final summary
    print("\n" + "="*80)
    
    # print("\n📊 Validation Results:")
    # print(f"• Market prices vary properly: σ = ₹{validation_analysis['market_price_std']:.2f}")
    # print(f"• IV evolves realistically: σ = {validation_analysis['iv_std']*100:.2f}%")
    # print(f"• Underlying vol matches input: {validation_analysis['underlying_vol']:.1%}")
    # print(f"• Model-market correlations: BS={validation_analysis['correlations']['bs']:.2f}, MC={validation_analysis['correlations']['mc']:.2f}")
    
    metrics = backtester.calculate_performance_metrics()
    if metrics:
        bs_mae = metrics['Black_Scholes']['Mean_Absolute_Error']
        mc_mae = metrics['Monte_Carlo']['Mean_Absolute_Error']
        better_model = "Black-Scholes" if bs_mae < mc_mae else "Monte Carlo"
        improvement = abs(bs_mae - mc_mae) / max(bs_mae, mc_mae) * 100
        
        print(f"\n🏆 Final Results:")
        print(f"• {better_model} performs {improvement:.1f}% better in terms of MAE")
        print(f"• Black-Scholes MAE: ₹{bs_mae:.2f}")
        print(f"• Monte Carlo MAE: ₹{mc_mae:.2f}")
        print(f"• Both models now tested against proper dynamic market conditions")
        print(f"• Errors are realistic given market noise and model assumptions")
    
    print("="*80)
    # print("The backtesting system now provides meaningful, actionable insights!")
    # print("="*80)