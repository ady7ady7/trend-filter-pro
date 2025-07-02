"""
Utilities Module
Helper functions for the Trend Filter Application
"""

import logging
import os
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import csv

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logger
    logger = logging.getLogger('trend_filter')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # File gets all messages
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {str(e)}")
    
    return logger

def save_results_to_csv(analysis_results: Dict[str, Any], filename: str) -> bool:
    """Save analysis results to CSV file"""
    
    try:
        # Flatten the nested dictionary structure
        flattened_data = flatten_analysis_results(analysis_results)
        
        # Create DataFrame
        df = pd.DataFrame([flattened_data])
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        return True
        
    except Exception as e:
        logger = logging.getLogger('trend_filter')
        logger.error(f"Error saving results to CSV: {str(e)}")
        return False

def flatten_analysis_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested analysis results for CSV export"""
    
    flattened = {}
    
    # Basic information
    flattened['symbol'] = results.get('symbol', '')
    flattened['analysis_timestamp'] = results.get('analysis_timestamp', '')
    flattened['trading_score'] = results.get('trading_score', 0)
    flattened['market_state'] = results.get('market_state', '')
    flattened['risk_assessment'] = results.get('risk_assessment', '')
    flattened['trend_direction'] = results.get('trend_direction', '')
    flattened['short_term_trend'] = results.get('short_term_trend', '')
    flattened['latest_price'] = results.get('latest_price', 0)
    flattened['price_change_24h'] = results.get('price_change_24h', 0)
    flattened['is_choppy_market'] = results.get('is_choppy_market', False)
    flattened['trend_strength'] = results.get('trend_strength', '')
    
    # Long-term analysis details
    long_term = results.get('long_term_analysis', {})
    if long_term:
        # EMA alignment
        ema_alignment = long_term.get('ema_alignment', {})
        flattened['ema_alignment_strength'] = ema_alignment.get('strength', 0)
        flattened['ema_alignment_type'] = ema_alignment.get('alignment', '')
        
        # Choppiness
        choppiness = long_term.get('choppiness', {})
        flattened['adx_value'] = choppiness.get('adx_value', 0)
        flattened['atr_value'] = choppiness.get('atr_value', 0)
        flattened['efficiency_ratio'] = choppiness.get('efficiency_ratio', 0)
        flattened['choppiness_score'] = choppiness.get('choppiness_score', 0)
        
        # Slopes
        slopes = long_term.get('slopes', {})
        for period, slope in slopes.items():
            flattened[f'long_term_{period}'] = slope
        
        flattened['data_points_long'] = long_term.get('data_points', 0)
    
    # Short-term analysis details
    short_term = results.get('short_term_analysis', {})
    if short_term:
        # Mean reversion
        mean_reversion = short_term.get('mean_reversion', {})
        flattened['rsi_value'] = mean_reversion.get('rsi_value', 0)
        flattened['rsi_signal'] = mean_reversion.get('rsi_signal', '')
        flattened['bb_position'] = mean_reversion.get('bb_position', 0)
        flattened['bb_signal'] = mean_reversion.get('bb_signal', '')
        flattened['distance_from_mean_pct'] = mean_reversion.get('distance_from_mean_pct', 0)
        
        # Recent crosses
        recent_crosses = short_term.get('recent_crosses', [])
        if recent_crosses:
            latest_cross = recent_crosses[-1]
            flattened['latest_cross_type'] = latest_cross.get('type', '')
            flattened['latest_cross_fast_ema'] = latest_cross.get('fast_ema', 0)
            flattened['latest_cross_slow_ema'] = latest_cross.get('slow_ema', 0)
            flattened['latest_cross_bars_ago'] = latest_cross.get('bars_ago', 0)
        else:
            flattened['latest_cross_type'] = ''
            flattened['latest_cross_fast_ema'] = 0
            flattened['latest_cross_slow_ema'] = 0
            flattened['latest_cross_bars_ago'] = 0
        
        flattened['data_points_short'] = short_term.get('data_points', 0)
    
    # Trading opportunities
    opportunities = results.get('opportunities', [])
    flattened['opportunities'] = '; '.join(opportunities) if opportunities else ''
    
    # Commentary
    flattened['commentary'] = results.get('commentary', '').replace('\n', ' | ')
    
    return flattened

def create_summary_report(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary report from multiple analysis results"""
    
    summary_data = []
    
    for result in results_list:
        summary_item = {
            'Symbol': result.get('symbol', ''),
            'Trading_Score': result.get('trading_score', 0),
            'Market_State': result.get('market_state', ''),
            'Trend_Direction': result.get('trend_direction', ''),
            'Risk_Level': result.get('risk_assessment', ''),
            'Latest_Price': result.get('latest_price', 0),
            'Price_Change_24h': result.get('price_change_24h', 0),
            'Is_Choppy': result.get('is_choppy_market', False),
            'ADX_Value': 0,
            'RSI_Value': 0,
            'Analysis_Time': result.get('analysis_timestamp', '')
        }
        
        # Extract ADX and RSI values
        long_term = result.get('long_term_analysis', {})
        if long_term:
            choppiness = long_term.get('choppiness', {})
            summary_item['ADX_Value'] = choppiness.get('adx_value', 0)
        
        short_term = result.get('short_term_analysis', {})
        if short_term:
            mean_reversion = short_term.get('mean_reversion', {})
            summary_item['RSI_Value'] = mean_reversion.get('rsi_value', 0)
        
        summary_data.append(summary_item)
    
    return pd.DataFrame(summary_data)

def format_price(price: float, symbol: str = "") -> str:
    """Format price based on instrument type"""
    
    if not price or price == 0:
        return "N/A"
    
    # Determine decimal places based on symbol
    if any(fx in symbol.upper() for fx in ['USD', 'EUR', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD']):
        # Forex pairs - typically 4-5 decimal places
        if 'JPY' in symbol.upper():
            return f"{price:.3f}"  # JPY pairs use 3 decimals
        else:
            return f"{price:.5f}"  # Other forex pairs use 5 decimals
    
    elif any(metal in symbol.upper() for metal in ['GC=F', 'SI=F', 'GOLD', 'SILVER']):
        # Precious metals - 2 decimal places
        return f"${price:.2f}"
    
    elif any(index in symbol.upper() for index in ['^GSPC', '^IXIC', '^DJI', '^GDAXI', '^FTSE']):
        # Stock indices - 2 decimal places
        return f"{price:.2f}"
    
    else:
        # Default formatting
        return f"{price:.2f}"

def calculate_performance_metrics(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate performance metrics for the analysis"""
    
    if not results_list:
        return {}
    
    total_instruments = len(results_list)
    
    # Count by trading score ranges
    excellent = sum(1 for r in results_list if r.get('trading_score', 0) >= 80)
    good = sum(1 for r in results_list if 60 <= r.get('trading_score', 0) < 80)
    fair = sum(1 for r in results_list if 40 <= r.get('trading_score', 0) < 60)
    poor = sum(1 for r in results_list if r.get('trading_score', 0) < 40)
    
    # Count by trend direction
    bullish = sum(1 for r in results_list if r.get('trend_direction', '') == 'bullish')
    bearish = sum(1 for r in results_list if r.get('trend_direction', '') == 'bearish')
    neutral = sum(1 for r in results_list if r.get('trend_direction', '') == 'neutral')
    
    # Count choppy markets
    choppy = sum(1 for r in results_list if r.get('is_choppy_market', False))
    
    # Risk levels
    low_risk = sum(1 for r in results_list if 'Low' in r.get('risk_assessment', ''))
    moderate_risk = sum(1 for r in results_list if 'Moderate' in r.get('risk_assessment', ''))
    high_risk = sum(1 for r in results_list if 'High' in r.get('risk_assessment', ''))
    
    # Average scores
    avg_score = sum(r.get('trading_score', 0) for r in results_list) / total_instruments
    avg_adx = 0
    avg_rsi = 0
    
    adx_values = []
    rsi_values = []
    
    for result in results_list:
        long_term = result.get('long_term_analysis', {})
        if long_term:
            choppiness = long_term.get('choppiness', {})
            adx_val = choppiness.get('adx_value', 0)
            if adx_val > 0:
                adx_values.append(adx_val)
        
        short_term = result.get('short_term_analysis', {})
        if short_term:
            mean_reversion = short_term.get('mean_reversion', {})
            rsi_val = mean_reversion.get('rsi_value', 0)
            if rsi_val > 0:
                rsi_values.append(rsi_val)
    
    if adx_values:
        avg_adx = sum(adx_values) / len(adx_values)
    if rsi_values:
        avg_rsi = sum(rsi_values) / len(rsi_values)
    
    return {
        'total_instruments': total_instruments,
        'score_distribution': {
            'excellent_80_plus': excellent,
            'good_60_79': good,
            'fair_40_59': fair,
            'poor_below_40': poor
        },
        'trend_distribution': {
            'bullish': bullish,
            'bearish': bearish,
            'neutral': neutral
        },
        'risk_distribution': {
            'low_risk': low_risk,
            'moderate_risk': moderate_risk,
            'high_risk': high_risk
        },
        'market_conditions': {
            'choppy_markets': choppy,
            'trending_markets': total_instruments - choppy
        },
        'averages': {
            'avg_trading_score': round(avg_score, 1),
            'avg_adx': round(avg_adx, 1),
            'avg_rsi': round(avg_rsi, 1)
        },
        'percentages': {
            'high_quality_setups': round((excellent / total_instruments) * 100, 1),
            'tradeable_opportunities': round(((excellent + good) / total_instruments) * 100, 1),
            'choppy_market_rate': round((choppy / total_instruments) * 100, 1)
        }
    }

def validate_data_quality(data: pd.DataFrame, symbol: str = "") -> Dict[str, Any]:
    """Validate the quality of market data"""
    
    quality_report = {
        'symbol': symbol,
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'data_points': len(data),
        'quality_score': 100
    }
    
    if data.empty:
        quality_report['is_valid'] = False
        quality_report['issues'].append("No data available")
        quality_report['quality_score'] = 0
        return quality_report
    
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        quality_report['is_valid'] = False
        quality_report['issues'].append(f"Missing columns: {missing_columns}")
        quality_report['quality_score'] -= 50
    
    # Check for NaN values
    nan_counts = data[required_columns].isnull().sum()
    total_nans = nan_counts.sum()
    
    if total_nans > 0:
        nan_percentage = (total_nans / (len(data) * len(required_columns))) * 100
        if nan_percentage > 10:
            quality_report['issues'].append(f"High NaN percentage: {nan_percentage:.1f}%")
            quality_report['quality_score'] -= 30
        elif nan_percentage > 5:
            quality_report['warnings'].append(f"Moderate NaN percentage: {nan_percentage:.1f}%")
            quality_report['quality_score'] -= 15
        else:
            quality_report['warnings'].append(f"Low NaN percentage: {nan_percentage:.1f}%")
            quality_report['quality_score'] -= 5
    
    # Check data consistency
    if len(data) > 0:
        # OHLC consistency
        invalid_ohlc = (
            (data['High'] < data[['Open', 'Close', 'Low']].max(axis=1)) |
            (data['Low'] > data[['Open', 'Close', 'High']].min(axis=1))
        ).sum()
        
        if invalid_ohlc > 0:
            invalid_percentage = (invalid_ohlc / len(data)) * 100
            if invalid_percentage > 5:
                quality_report['issues'].append(f"Invalid OHLC relationships: {invalid_percentage:.1f}%")
                quality_report['quality_score'] -= 25
            else:
                quality_report['warnings'].append(f"Some invalid OHLC relationships: {invalid_percentage:.1f}%")
                quality_report['quality_score'] -= 10
    
    # Check for gaps in data
    if len(data) > 1:
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_interval = time_diffs.median()
            large_gaps = (time_diffs > median_interval * 3).sum()
            
            if large_gaps > len(data) * 0.1:  # More than 10% large gaps
                quality_report['warnings'].append(f"Multiple data gaps detected: {large_gaps}")
                quality_report['quality_score'] -= 10
    
    # Check minimum data requirements
    if len(data) < 50:
        quality_report['warnings'].append("Insufficient data for reliable analysis (< 50 points)")
        quality_report['quality_score'] -= 20
    
    # Final validation
    if quality_report['quality_score'] < 50:
        quality_report['is_valid'] = False
    
    return quality_report

def export_detailed_report(results: Dict[str, Any], filename: str) -> bool:
    """Export detailed analysis report to text file"""
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"DETAILED TREND ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic information
            f.write(f"INSTRUMENT: {results.get('symbol', 'Unknown')}\n")
            f.write(f"Latest Price: {results.get('latest_price', 0)}\n")
            f.write(f"24h Change: {results.get('price_change_24h', 0):.2f}%\n")
            f.write(f"Trading Score: {results.get('trading_score', 0)}/100\n")
            f.write(f"Market State: {results.get('market_state', 'Unknown')}\n")
            f.write(f"Risk Assessment: {results.get('risk_assessment', 'Unknown')}\n\n")
            
            # Commentary
            f.write("ANALYSIS COMMENTARY:\n")
            f.write("-" * 40 + "\n")
            commentary = results.get('commentary', 'No commentary available')
            f.write(commentary + "\n\n")
            
            # Trading opportunities
            f.write("TRADING OPPORTUNITIES:\n")
            f.write("-" * 40 + "\n")
            opportunities = results.get('opportunities', [])
            for i, opp in enumerate(opportunities, 1):
                f.write(f"{i}. {opp}\n")
            f.write("\n")
            
            # Detailed technical analysis
            f.write("TECHNICAL ANALYSIS DETAILS:\n")
            f.write("-" * 40 + "\n")
            
            # Long-term analysis
            long_term = results.get('long_term_analysis', {})
            if long_term:
                f.write("Long-term Analysis:\n")
                f.write(f"  Trend Direction: {long_term.get('trend_direction', 'Unknown')}\n")
                
                ema_alignment = long_term.get('ema_alignment', {})
                f.write(f"  EMA Alignment: {ema_alignment.get('alignment', 'Unknown')} ({ema_alignment.get('strength', 0):.1f}%)\n")
                
                choppiness = long_term.get('choppiness', {})
                f.write(f"  ADX Value: {choppiness.get('adx_value', 0):.1f}\n")
                f.write(f"  Trend Strength: {choppiness.get('trend_strength', 'Unknown')}\n")
                f.write(f"  Is Choppy: {choppiness.get('is_choppy', False)}\n\n")
            
            # Short-term analysis
            short_term = results.get('short_term_analysis', {})
            if short_term:
                f.write("Short-term Analysis:\n")
                f.write(f"  Trend Direction: {short_term.get('trend_direction', 'Unknown')}\n")
                
                mean_reversion = short_term.get('mean_reversion', {})
                f.write(f"  RSI: {mean_reversion.get('rsi_value', 0):.1f} ({mean_reversion.get('rsi_signal', 'Unknown')})\n")
                f.write(f"  BB Position: {mean_reversion.get('bb_position', 0):.2f} ({mean_reversion.get('bb_signal', 'Unknown')})\n")
                
                recent_crosses = short_term.get('recent_crosses', [])
                if recent_crosses:
                    f.write("  Recent EMA Crosses:\n")
                    for cross in recent_crosses[-3:]:  # Last 3 crosses
                        f.write(f"    {cross['type'].title()} cross: EMA{cross['fast_ema']}/EMA{cross['slow_ema']} ({cross['bars_ago']} bars ago)\n")
        
        return True
        
    except Exception as e:
        logger = logging.getLogger('trend_filter')
        logger.error(f"Error exporting detailed report: {str(e)}")
        return False

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    
    import platform
    import sys
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }