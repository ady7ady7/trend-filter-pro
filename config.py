"""
Configuration Module
Central configuration for the Trend Filter Application
"""

import os
from datetime import datetime

class Config:
    """Application configuration settings"""
    
    def __init__(self):
        
        # Data fetching parameters
        self.LONG_TERM_PERIOD = "3mo"      # 3 months for long-term analysis
        self.LONG_TERM_INTERVAL = "1h"     # 1-hour intervals
        
        self.SHORT_TERM_PERIOD = "5d"      # 5 days for short-term analysis  
        self.SHORT_TERM_INTERVAL = "1m"    # 1-minute intervals
        
        # Alternative configurations for different data availability
        self.BACKUP_CONFIGS = [
            {
                "long_term": {"period": "3mo", "interval": "15m"},
                "short_term": {"period": "3d", "interval": "1m"}
            },
            {
                "long_term": {"period": "1mo", "interval": "30m"},
                "short_term": {"period": "2d", "interval": "2m"}
            }
        ]
        
        # Technical analysis parameters
        self.EMA_PERIODS = [8, 13, 21, 34, 55, 89, 144]  # Fibonacci sequence
        self.RSI_PERIOD = 14
        self.BB_PERIOD = 20
        self.BB_STD = 2
        self.ADX_PERIOD = 14
        self.ATR_PERIOD = 14
        
        # Trend analysis thresholds
        self.SLOPE_THRESHOLD = 0.05          # Minimum slope for trend (0.05%)
        self.TREND_STRENGTH_THRESHOLD = 25   # ADX threshold for trend strength
        self.CHOPPINESS_THRESHOLD = 50       # Choppiness score threshold
        self.ALIGNMENT_THRESHOLD = 0.6       # EMA alignment threshold
        
        # Trading score weights
        self.SCORE_WEIGHTS = {
            'trend_alignment': 0.25,
            'timeframe_agreement': 0.20,
            'trend_strength': 0.20,
            'choppiness_penalty': 0.15,
            'mean_reversion': 0.10,
            'momentum': 0.10
        }
        
        # Risk assessment parameters
        self.RISK_FACTORS = {
            'high_choppiness': 3,
            'conflicting_timeframes': 2,
            'weak_trend': 2,
            'mixed_alignment': 1,
            'high_volatility': 1
        }
        
        # Default instruments for batch processing
        self.DEFAULT_INSTRUMENTS = [
            # Major Forex Pairs
            "EURUSD=X",
            "GBPUSD=X", 
            "USDJPY=X",
            "USDCHF=X",
            "AUDUSD=X",
            "USDCAD=X",
            
            # Major US Indices
            "^GSPC",    # S&P 500
            "^IXIC",    # NASDAQ
            "^DJI",     # Dow Jones
            
            # European Indices
            "^GDAXI",   # DAX
            "^FTSE",    # FTSE 100
            
            # Precious Metals
            "GC=F",     # Gold
            "SI=F",     # Silver
            
            # Commodities
            "CL=F",     # Crude Oil
        ]
        
        # Output settings
        self.OUTPUT_DIR = "trend_analysis_results"
        self.CSV_COLUMNS = [
            'symbol',
            'analysis_timestamp', 
            'trading_score',
            'market_state',
            'risk_assessment',
            'trend_direction',
            'short_term_trend',
            'latest_price',
            'price_change_24h',
            'is_choppy_market',
            'trend_strength',
            'ema_alignment_strength',
            'ema_alignment_type',
            'adx_value',
            'rsi_value',
            'rsi_signal',
            'bb_position',
            'bb_signal',
            'recent_crosses',
            'opportunities',
            'commentary',
            'data_points_long',
            'data_points_short'
        ]
        
        # Logging configuration
        self.LOG_LEVEL = "INFO"
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.LOG_FILE = f"trend_filter_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Application metadata
        self.APP_NAME = "Trend Filter Pro"
        self.VERSION = "1.0.0"
        self.AUTHOR = "Trading Analysis System"
        
        # Ensure output directory exists
        self._create_output_directory()
    
    def _create_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
    
    def get_timeframe_config(self, config_index: int = 0) -> dict:
        """Get timeframe configuration by index"""
        if 0 <= config_index < len(self.BACKUP_CONFIGS):
            return self.BACKUP_CONFIGS[config_index]
        else:
            return {
                "long_term": {"period": self.LONG_TERM_PERIOD, "interval": self.LONG_TERM_INTERVAL},
                "short_term": {"period": self.SHORT_TERM_PERIOD, "interval": self.SHORT_TERM_INTERVAL}
            }
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        
        # Data fetching
        self.LONG_TERM_PERIOD = os.getenv('LONG_TERM_PERIOD', self.LONG_TERM_PERIOD)
        self.LONG_TERM_INTERVAL = os.getenv('LONG_TERM_INTERVAL', self.LONG_TERM_INTERVAL)
        self.SHORT_TERM_PERIOD = os.getenv('SHORT_TERM_PERIOD', self.SHORT_TERM_PERIOD)  
        self.SHORT_TERM_INTERVAL = os.getenv('SHORT_TERM_INTERVAL', self.SHORT_TERM_INTERVAL)
        
        # Thresholds
        if os.getenv('SLOPE_THRESHOLD'):
            self.SLOPE_THRESHOLD = float(os.getenv('SLOPE_THRESHOLD'))
        
        if os.getenv('TREND_STRENGTH_THRESHOLD'):
            self.TREND_STRENGTH_THRESHOLD = float(os.getenv('TREND_STRENGTH_THRESHOLD'))
        
        if os.getenv('CHOPPINESS_THRESHOLD'):
            self.CHOPPINESS_THRESHOLD = float(os.getenv('CHOPPINESS_THRESHOLD'))
        
        # Output directory
        self.OUTPUT_DIR = os.getenv('OUTPUT_DIR', self.OUTPUT_DIR)
        
        # Recreate output directory with new path
        self._create_output_directory()
    
    def get_symbol_specific_config(self, symbol: str) -> dict:
        """Get symbol-specific configuration overrides"""
        
        # Default configuration
        config = {
            'ema_periods': self.EMA_PERIODS,
            'slope_threshold': self.SLOPE_THRESHOLD,
            'trend_strength_threshold': self.TREND_STRENGTH_THRESHOLD
        }
        
        # Symbol-specific overrides
        symbol_configs = {
            # Forex pairs - typically more sensitive
            'EURUSD=X': {
                'slope_threshold': 0.03,
                'trend_strength_threshold': 20
            },
            'GBPUSD=X': {
                'slope_threshold': 0.04,
                'trend_strength_threshold': 22
            },
            
            # Indices - less sensitive to small moves
            '^GSPC': {
                'slope_threshold': 0.08,
                'trend_strength_threshold': 30
            },
            '^GDAXI': {
                'slope_threshold': 0.10,
                'trend_strength_threshold': 28
            },
            
            # Commodities - can be very volatile
            'GC=F': {
                'slope_threshold': 0.06,
                'trend_strength_threshold': 25,
                'ema_periods': [8, 13, 21, 34, 55]  # Fewer EMAs for faster signals
            },
            'CL=F': {
                'slope_threshold': 0.12,
                'trend_strength_threshold': 35
            }
        }
        
        # Apply symbol-specific overrides
        if symbol in symbol_configs:
            config.update(symbol_configs[symbol])
        
        return config
    
    def validate_configuration(self) -> list:
        """Validate configuration settings and return any issues"""
        issues = []
        
        # Validate periods
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if self.LONG_TERM_PERIOD not in valid_periods:
            issues.append(f"Invalid LONG_TERM_PERIOD: {self.LONG_TERM_PERIOD}")
        
        if self.SHORT_TERM_PERIOD not in valid_periods:
            issues.append(f"Invalid SHORT_TERM_PERIOD: {self.SHORT_TERM_PERIOD}")
        
        # Validate intervals
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if self.LONG_TERM_INTERVAL not in valid_intervals:
            issues.append(f"Invalid LONG_TERM_INTERVAL: {self.LONG_TERM_INTERVAL}")
            
        if self.SHORT_TERM_INTERVAL not in valid_intervals:
            issues.append(f"Invalid SHORT_TERM_INTERVAL: {self.SHORT_TERM_INTERVAL}")
        
        # Validate EMA periods
        if not self.EMA_PERIODS or len(self.EMA_PERIODS) < 3:
            issues.append("EMA_PERIODS should contain at least 3 periods")
        
        # Validate thresholds
        if not 0 < self.SLOPE_THRESHOLD < 1:
            issues.append("SLOPE_THRESHOLD should be between 0 and 1")
            
        if not 0 < self.TREND_STRENGTH_THRESHOLD < 100:
            issues.append("TREND_STRENGTH_THRESHOLD should be between 0 and 100")
        
        # Validate output directory
        if not os.path.exists(self.OUTPUT_DIR):
            try:
                os.makedirs(self.OUTPUT_DIR)
            except Exception as e:
                issues.append(f"Cannot create output directory: {str(e)}")
        
        return issues
    
    def print_configuration(self):
        """Print current configuration"""
        print(f"\n{'='*50}")
        print(f"{self.APP_NAME} v{self.VERSION} - Configuration")
        print(f"{'='*50}")
        print(f"Long-term Analysis: {self.LONG_TERM_PERIOD} @ {self.LONG_TERM_INTERVAL}")
        print(f"Short-term Analysis: {self.SHORT_TERM_PERIOD} @ {self.SHORT_TERM_INTERVAL}")
        print(f"EMA Periods: {self.EMA_PERIODS}")
        print(f"Slope Threshold: {self.SLOPE_THRESHOLD}%")
        print(f"Trend Strength Threshold: {self.TREND_STRENGTH_THRESHOLD}")
        print(f"Output Directory: {self.OUTPUT_DIR}")
        print(f"Default Instruments: {len(self.DEFAULT_INSTRUMENTS)} symbols")
        print(f"{'='*50}\n")