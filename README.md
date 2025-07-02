# Trend Filter Pro - Trading Analysis Application

A comprehensive Python application for analyzing trading instruments and identifying trend opportunities across multiple timeframes.

## Features

- **Multi-timeframe Analysis**: Long-term (3 months, 1h intervals) and short-term (5 days, 5m intervals) trend analysis
- **Comprehensive Technical Analysis**: EMA alignment, slope analysis, momentum indicators, mean reversion signals
- **Smart Trend Detection**: Identifies trending vs choppy markets to avoid poor trading conditions
- **Risk Assessment**: Scores trading opportunities from 0-100 with detailed risk analysis
- **Batch Processing**: Analyze multiple instruments automatically
- **CSV Export**: All results saved in convenient CSV format for further analysis
- **Detailed Reporting**: Comprehensive commentary and trading recommendations

## Supported Instruments

- **Forex Pairs**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- **Major Indices**: S&P 500, NASDAQ, Dow Jones, DAX, FTSE 100, Nikkei 225
- **Precious Metals**: Gold, Silver
- **Commodities**: Crude Oil, Brent Oil

## Installation

### Prerequisites

- Python 3.8 or higher
- VS Code (recommended) or any Python IDE

### Step 1: Clone or Download Files

Create a new folder for your project and save all the Python files:

```
trend_filter_app/
├── main.py
├── trend_analyzer.py
├── data_manager.py
├── config.py
├── utils.py
└── requirements.txt
```

### Step 2: Install TA-Lib

TA-Lib is required for technical analysis. Installation varies by platform:

#### Windows:
```bash
# Download TA-Lib wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Example for Python 3.9, 64-bit:
pip install TA_Lib‑0.4.25‑cp39‑cp39‑win_amd64.whl
```

#### macOS:
```bash
brew install ta-lib
pip install TA-Lib
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

Test the installation by running:

```bash
python -c "import yfinance, talib, pandas, numpy; print('All dependencies installed successfully!')"
```

## Usage

### Quick Start

1. **Open VS Code** and navigate to your project folder
2. **Run the application**:
   ```bash
   python main.py
   ```
3. **Choose analysis mode**:
   - Option 1: Analyze single instrument
   - Option 2: Analyze multiple instruments from predefined list

### Single Instrument Analysis

```bash
python main.py
# Select option 1
# Enter symbol: EURUSD=X
```

### Batch Analysis

```bash
python main.py
# Select option 2
# Choose to use default list or enter custom symbols
```

### Symbol Format Examples

- **Forex**: `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`
- **Indices**: `^GSPC` (S&P 500), `^IXIC` (NASDAQ), `^GDAXI` (DAX)
- **Metals**: `GC=F` (Gold), `SI=F` (Silver)
- **Oil**: `CL=F` (Crude Oil), `BZ=F` (Brent)

## Output Files

The application generates several types of output files:

### Individual Analysis Files
- **Format**: `trend_analysis_[SYMBOL]_[TIMESTAMP].csv`
- **Contains**: Complete analysis with all technical indicators, scores, and commentary

### Summary Report (for batch analysis)
- **Format**: `trend_analysis_summary_[TIMESTAMP].csv`
- **Contains**: Overview of all analyzed instruments with key metrics

### Log Files
- **Format**: `trend_filter_[DATE].log`
- **Contains**: Application logs for debugging and monitoring

## Understanding the Results

### Trading Score (0-100)
- **80-100**: Excellent opportunity - High probability setup
- **60-79**: Good opportunity - Favorable conditions
- **40-59**: Neutral - Wait for better confirmation
- **0-39**: Avoid - Poor trading conditions

### Market States
- **Strong [Bullish/Bearish] Trend**: High confidence trading opportunity
- **Moderate [Bullish/Bearish] Trend**: Good opportunity with confirmation
- **Choppy/Ranging Market**: Avoid trading
- **Weak Trend**: Low conviction, wait for better setup

### Risk Levels
- **Low Risk**: Favorable conditions, aligned timeframes
- **Moderate Risk**: Some conflicting signals
- **High Risk**: Choppy market or strong conflicting signals

## Configuration

### Timeframe Settings

Edit `config.py` to customize analysis periods:

```python
# Long-term analysis
LONG_TERM_PERIOD = "3mo"    # 3 months
LONG_TERM_INTERVAL = "1h"   # 1-hour intervals

# Short-term analysis  
SHORT_TERM_PERIOD = "5d"    # 5 days
SHORT_TERM_INTERVAL = "1m"  # 5-minute intervals
```

### Technical Analysis Parameters

```python
# EMA periods (Fibonacci sequence)
EMA_PERIODS = [8, 13, 21, 34, 55, 89, 144]

# Trend strength thresholds
SLOPE_THRESHOLD = 0.05          # Minimum slope for trend
TREND_STRENGTH_THRESHOLD = 25   # ADX threshold
```

### Adding Custom Instruments

Edit the `DEFAULT_INSTRUMENTS` list in `config.py`:

```python
DEFAULT_INSTRUMENTS = [
    "EURUSD=X",
    "^GSPC", 
    "GC=F",
    # Add your symbols here
]
```

## Analysis Methodology

### 1. Data Collection
- Fetches 3 months of hourly data for long-term trend analysis
- Fetches 5 days of 5-minute data for short-term precision
- Validates data quality and handles missing values

### 2. Trend Analysis
- **EMA Alignment**: Analyzes positioning of multiple EMAs
- **Slope Calculation**: Measures trend strength and direction
- **Cross Detection**: Identifies recent EMA crossovers
- **Momentum**: Evaluates trend acceleration/deceleration

### 3. Mean Reversion Analysis
- **RSI**: Identifies overbought/oversold conditions
- **Bollinger Bands**: Measures price position relative to volatility bands
- **Distance from Mean**: Calculates deviation from 20-period EMA

### 4. Market Condition Assessment
- **ADX**: Measures trend strength vs ranging conditions
- **Efficiency Ratio**: Evaluates price movement efficiency
- **Volatility Analysis**: Compares recent vs historical volatility

### 5. Scoring Algorithm
Combines multiple factors with weighted scoring:
- Trend alignment strength (25%)
- Timeframe agreement (20%)
- Trend strength (20%)
- Choppiness penalty (15%)
- Mean reversion opportunity (10%)
- Momentum confirmation (10%)

## Troubleshooting

### Common Issues

#### 1. TA-Lib Installation Problems
```bash
# Windows: Download wheel file manually
# macOS: Install Homebrew first
# Linux: Install system dependencies
```

#### 2. Symbol Not Found
- Verify symbol format (use Yahoo Finance format)
- Check if instrument is available on Yahoo Finance
- Try alternative symbol naming

#### 3. No Data Retrieved
- Check internet connection
- Verify symbol exists and is tradeable
- Some symbols may have limited historical data

#### 4. Import Errors
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Data Limitations

- **Minute data**: Limited to last 7-60 days depending on interval
- **Hourly data**: Available for longer periods (2+ years)
- **Weekend/Holiday gaps**: Normal for traditional markets
- **Crypto**: 24/7 data available
- **Forex**: 24/5 data (closed weekends)

## Advanced Usage

### Environment Variables

Set environment variables to customize behavior:

```bash
export LONG_TERM_PERIOD="6mo"
export SHORT_TERM_PERIOD="7d"
export OUTPUT_DIR="custom_results"
```

### Custom Symbol Mapping

Add custom symbols in `data_manager.py`:

```python
self.symbol_mapping = {
    'CUSTOM_SYMBOL': 'YAHOO_FORMAT',
    # Add your mappings
}
```

### Logging Configuration

Adjust logging level in `utils.py`:

```python
logger = setup_logging(log_level="DEBUG")  # For detailed logs
```

## Integration Ideas

### Excel Integration
- Import CSV files into Excel for advanced charting
- Create custom dashboards and alerts
- Combine with fundamental analysis

### Trading Platform Integration
- Use results to inform manual trading decisions
- Export signals for automated trading systems
- Combine with position sizing algorithms

### Portfolio Management
- Analyze entire portfolio for trend opportunities
- Diversification analysis across asset classes
- Risk-adjusted position sizing

## Disclaimer

This application is for educational and analysis purposes only. It does not constitute financial advice. Always perform your own analysis and risk management before making trading decisions.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Yahoo Finance symbol formats
3. Verify all dependencies are installed correctly
4. Check log files for detailed error information