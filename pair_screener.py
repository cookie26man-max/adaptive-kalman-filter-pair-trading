import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

class CointegrationScreener:
    """
    Automated statistical screening engine to identify valid, 
    stationary pairs for the Kalman Filter Trading Loop.
    """
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()

    def download_data(self) -> None:
        print(f"Downloading historical data for {len(self.tickers)} assets...")
        raw = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
        # Safely extract closing prices across different yfinance index formats
        if 'Close' in raw.columns.levels[0]:
            self.data = raw['Close'].ffill().bfill()
        else:
            self.data = raw.ffill().bfill()
            
    def screen_pairs(self, p_value_threshold: float = 0.05) -> pd.DataFrame:
        """
        Tests all unique combinations using log-prices and an ADF test on the residuals.
        """
        if self.data.empty:
            self.download_data()

        columns = self.data.columns
        n = len(columns)
        pairs_results = []

        print("Executing statistical screening matrix...")
        # Loop through unique combinations (avoiding checking a pair twice or against itself)
        for i in range(n):
            for j in range(i + 1, n):
                asset_y = columns[i]
                asset_x = columns[j]
                
                # Operate purely on log-prices to match our scale-invariant engine
                y_log = np.log(self.data[asset_y])
                x_log = np.log(self.data[asset_x])
                
                # Run a quick OLS to extract static baseline residuals for the ADF test
                matrix_x = np.vstack([np.ones(len(x_log)), x_log]).T
                beta_static, alpha_static = np.linalg.lstsq(matrix_x, y_log, rcond=None)[0]
                residuals = y_log - (beta_static * x_log + alpha_static)
                
                # Run the Augmented Dickey-Fuller test
                try:
                    adf_result = adfuller(residuals, maxlag=1)
                    p_value = adf_result[1]
                    t_stat = adf_result[0]
                    critical_value_5pct = adf_result[4]['5%']
                    
                    # If the p-value passes the threshold, it is a statistically valid pair
                    if p_value < p_value_threshold:
                        pairs_results.append({
                            'Asset_Y': asset_y,
                            'Asset_X': asset_x,
                            'P-Value': p_value,
                            'T-Stat': t_stat,
                            'Crit_5%': critical_value_5pct,
                            'Static_Beta': beta_static
                        })
                except Exception:
                    continue

        df_pairs = pd.DataFrame(pairs_results)
        if not df_pairs.empty:
            # Sort by the strongest cointegration (lowest p-value)
            return df_pairs.sort_values(by='P-Value').reset_index(drop=True)
        return pd.DataFrame()

if __name__ == "__main__":
    # Test basket: Tech and Consumer Mega-Caps
    ticker_basket = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "KO", "PEP", "XOM", "CVX"]
    
    screener = CointegrationScreener(ticker_basket, start_date="2024-01-01", end_date="2026-01-01")
    valid_pairs = screener.screen_pairs(p_value_threshold=0.05)
    
    print("\n================ TOP COINTEGRATED PAIRS FOUND ================")
    if not valid_pairs.empty:
        print(valid_pairs.to_string(index=False))
    else:
        print("No statistically valid cointegrated pairs found in this time horizon.")
    print("==============================================================")
