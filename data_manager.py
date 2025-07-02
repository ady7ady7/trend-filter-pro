"""
Data Manager Module
Handles fetching and preprocessing of financial data using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

class DataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Symbol mapping for common instruments
        self.symbol_mapping = {
            # Major Forex Pairs
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X',
            
            # Major Indices
            'SPX': '^GSPC',      # S&P 500
            'NDX': '^IXIC',      # NASDAQ
            'DJI': '^DJI',       # Dow Jones
            'DAX': '^GDAXI',     # German DAX
            'FTSE': '^FTSE',     # FTSE 100
            'NIKKEI': '^N225',   # Nikkei 225
            
            # Precious Metals
            'GOLD': 'GC=F',      # Gold Futures
            'SILVER': 'SI=F',    # Silver Futures
            'XAUUSD': 'GC=F',    # Gold alternative
            'XAGUSD': 'SI=F',    # Silver alternative
            
            # Commodities
            'OIL': 'CL=F',       # Crude Oil
            'BRENT': 'BZ=F',     # Brent Oil
            
            # Crypto (if available)
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD'
        }
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for yfinance compatibility"""
        symbol = symbol.upper().strip()
        
        # Check if symbol needs mapping
        if symbol in self.symbol_mapping:
            return self.symbol_mapping[symbol]
        
        # If already properly formatted, return as is
        return symbol
    
    def fetch_data(self, symbol: str, period: str = "3mo", interval: str = "1h") -> pd.DataFrame:
        """
        Fetch market data using yfinance
        
        Args:
            symbol: Trading symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        
        try:
            # Normalize symbol
            normalized_symbol = self.normalize_symbol(symbol)
            
            self.logger.info(f"Fetching data for {normalized_symbol} (period: {period}, interval: {interval})")
            
            # Create ticker object
            ticker = yf.Ticker(normalized_symbol)
            
            # Fetch data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.error(f"No data returned for symbol {normalized_symbol}")
                return pd.DataFrame()
            
            # Clean and validate data
            data = self.clean_data(data)
            
            self.logger.info(f"Successfully fetched {len(data)} data points for {normalized_symbol}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data"""
        
        if data.empty:
            return data
        
        # Ensure we have required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"Missing column: {col}")
                return pd.DataFrame()
        
        # Remove rows with NaN values in OHLC
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        data = data.dropna(subset=ohlc_columns)
        
        # Validate OHLC relationships
        # High should be >= Open, Close, Low
        # Low should be <= Open, Close, High
        invalid_rows = (
            (data['High'] < data[['Open', 'Close', 'Low']].max(axis=1)) |
            (data['Low'] > data[['Open', 'Close', 'High']].min(axis=1))
        )
        
        if invalid_rows.any():
            self.logger.warning(f"Found {invalid_rows.sum()} invalid OHLC rows, removing them")
            data = data[~invalid_rows]
        
        # Remove duplicate timestamps
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Add basic validation metrics
        data['Price_Range'] = data['High'] - data['Low']
        data['Body_Size'] = abs(data['Close'] - data['Open'])
        
        return data
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic information about the trading symbol"""
        
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            
            # Get basic info
            info = ticker.info
            
            # Extract relevant information
            symbol_info = {
                'symbol': normalized_symbol,
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'currency': info.get('currency', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'timezone': info.get('timeZoneFullName', 'Unknown')
            }
            
            return symbol_info
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'name': 'Unknown',
                'currency': 'Unknown',
                'exchange': 'Unknown'
            }
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and has data"""
        
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            
            # Try to fetch 1 day of data
            test_data = ticker.history(period="1d", interval="1h")
            
            return not test_data.empty
            
        except Exception as e:
            self.logger.error(f"Symbol validation failed for {symbol}: {str(e)}")
            return False
    
    def get_available_periods_intervals(self) -> Dict[str, Any]:
        """Get available periods and intervals for yfinance"""
        
        return {
            'periods': {
                '1d': '1 day',
                '5d': '5 days', 
                '1mo': '1 month',
                '3mo': '3 months',
                '6mo': '6 months',
                '1y': '1 year',
                '2y': '2 years',
                '5y': '5 years',
                '10y': '10 years',
                'ytd': 'Year to date',
                'max': 'Maximum available'
            },
            'intervals': {
                '1m': '1 minute',
                '2m': '2 minutes',
                '5m': '5 minutes', 
                '15m': '15 minutes',
                '30m': '30 minutes',
                '60m': '60 minutes',
                '90m': '90 minutes',
                '1h': '1 hour',
                '1d': '1 day',
                '5d': '5 days',
                '1wk': '1 week',
                '1mo': '1 month',
                '3mo': '3 months'
            },
            'limitations': {
                '1m': 'Limited to 7 days',
                '2m': 'Limited to 60 days',
                '5m': 'Limited to 60 days',
                '15m': 'Limited to 60 days',
                '30m': 'Limited to 60 days',
                '60m': 'Limited to 730 days',
                '90m': 'Limited to 60 days'
            }
        }
    
    def fetch_multiple_timeframes(self, symbol: str, timeframes: Dict[str, Dict[str, str]]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes
        
        Args:
            symbol: Trading symbol
            timeframes: Dict with timeframe names as keys and {'period': str, 'interval': str} as values
        
        Returns:
            Dict with timeframe names as keys and DataFrames as values
        """
        
        results = {}
        
        for timeframe_name, params in timeframes.items():
            try:
                data = self.fetch_data(
                    symbol=symbol,
                    period=params['period'],
                    interval=params['interval']
                )
                
                if not data.empty:
                    results[timeframe_name] = data
                    self.logger.info(f"Successfully fetched {timeframe_name} data: {len(data)} points")
                else:
                    self.logger.warning(f"No data for {timeframe_name}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching {timeframe_name} data: {str(e)}")
        
        return results
    
    def calculate_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for the dataset"""
        
        if data.empty:
            return {}
        
        try:
            stats = {
                'data_points': len(data),
                'date_range': {
                    'start': data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_hours': (data.index[-1] - data.index[0]).total_seconds() / 3600
                },
                'price_stats': {
                    'current_price': float(data['Close'].iloc[-1]),
                    'highest_price': float(data['High'].max()),
                    'lowest_price': float(data['Low'].min()),
                    'average_price': float(data['Close'].mean()),
                    'price_change': float(data['Close'].iloc[-1] - data['Close'].iloc[0]),
                    'price_change_pct': float(((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100)
                },
                'volume_stats': {
                    'total_volume': float(data['Volume'].sum()),
                    'average_volume': float(data['Volume'].mean()),
                    'max_volume': float(data['Volume'].max())
                },
                'volatility': {
                    'daily_returns_std': float(data['Close'].pct_change().std() * np.sqrt(252)),  # Annualized
                    'average_range': float((data['High'] - data['Low']).mean()),
                    'average_range_pct': float(((data['High'] - data['Low']) / data['Close']).mean() * 100)
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating basic stats: {str(e)}")
            return {}