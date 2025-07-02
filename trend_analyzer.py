"""
Trend Analyzer Module
Comprehensive trend analysis using pandas-ta (no TA-Lib dependency!)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class TrendAnalyzer:
    def __init__(self):
        self.ema_periods = [8, 13, 21, 34, 55, 89, 144]  # Fibonacci-based EMAs
        self.trend_strength_threshold = 0.001  # 0.1% minimum slope
        
    def calculate_emas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple EMAs for trend analysis"""
        df = data.copy()
        
        for period in self.ema_periods:
            df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)
            
        return df
    
    def calculate_slopes(self, data: pd.DataFrame, lookback: int = 10) -> Dict[str, float]:
        """Calculate slopes for different EMAs"""
        slopes = {}
        
        for period in self.ema_periods:
            ema_col = f'EMA_{period}'
            if ema_col in data.columns and len(data) >= lookback:
                # Calculate slope using linear regression over lookback period
                recent_data = data[ema_col].tail(lookback).values
                if not pd.isna(recent_data).any():
                    x = np.arange(len(recent_data))
                    slope = np.polyfit(x, recent_data, 1)[0]
                    # Normalize slope as percentage change per period
                    slopes[f'slope_{period}'] = (slope / recent_data[-1]) * 100 if recent_data[-1] != 0 else 0
                else:
                    slopes[f'slope_{period}'] = 0
        
        return slopes
    
    def analyze_ema_alignment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze EMA alignment for trend strength"""
        latest_row = data.iloc[-1]
        
        # Get EMA values in order
        ema_values = []
        for period in sorted(self.ema_periods):
            ema_col = f'EMA_{period}'
            if ema_col in latest_row and not pd.isna(latest_row[ema_col]):
                ema_values.append((period, latest_row[ema_col]))
        
        if len(ema_values) < 3:
            return {'alignment': 'insufficient_data', 'strength': 0}
        
        # Check for bullish alignment (shorter EMAs above longer EMAs)
        bullish_aligned = 0
        bearish_aligned = 0
        
        for i in range(len(ema_values) - 1):
            if ema_values[i][1] > ema_values[i + 1][1]:  # Shorter EMA > Longer EMA
                bullish_aligned += 1
            elif ema_values[i][1] < ema_values[i + 1][1]:  # Shorter EMA < Longer EMA
                bearish_aligned += 1
        
        total_comparisons = len(ema_values) - 1
        bullish_ratio = bullish_aligned / total_comparisons if total_comparisons > 0 else 0
        bearish_ratio = bearish_aligned / total_comparisons if total_comparisons > 0 else 0
        
        # Determine alignment
        if bullish_ratio >= 0.8:
            alignment = 'strong_bullish'
        elif bullish_ratio >= 0.6:
            alignment = 'bullish'
        elif bearish_ratio >= 0.8:
            alignment = 'strong_bearish'
        elif bearish_ratio >= 0.6:
            alignment = 'bearish'
        else:
            alignment = 'mixed'
        
        # Calculate alignment strength (0-100)
        strength = max(bullish_ratio, bearish_ratio) * 100
        
        return {
            'alignment': alignment,
            'strength': strength,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'ema_values': ema_values
        }
    
    def detect_ema_crosses(self, data: pd.DataFrame, lookback: int = 20) -> List[Dict]:
        """Detect recent EMA crossovers"""
        crosses = []
        
        if len(data) < lookback:
            return crosses
        
        recent_data = data.tail(lookback)
        
        # Check major EMA pairs for crosses
        ema_pairs = [
            (8, 21), (13, 21), (21, 55), (34, 89), (55, 144)
        ]
        
        for fast_period, slow_period in ema_pairs:
            fast_col = f'EMA_{fast_period}'
            slow_col = f'EMA_{slow_period}'
            
            if fast_col in recent_data.columns and slow_col in recent_data.columns:
                fast_ema = recent_data[fast_col]
                slow_ema = recent_data[slow_col]
                
                # Remove NaN values
                valid_mask = ~(pd.isna(fast_ema) | pd.isna(slow_ema))
                if valid_mask.sum() < 2:
                    continue
                
                fast_ema = fast_ema[valid_mask]
                slow_ema = slow_ema[valid_mask]
                
                # Find crossovers
                fast_above = fast_ema > slow_ema
                crossovers = fast_above != fast_above.shift(1)
                
                for idx in crossovers[crossovers].index:
                    cross_type = 'bullish' if fast_above.loc[idx] else 'bearish'
                    
                    crosses.append({
                        'date': idx,
                        'type': cross_type,
                        'fast_ema': fast_period,
                        'slow_ema': slow_period,
                        'fast_value': fast_ema.loc[idx],
                        'slow_value': slow_ema.loc[idx],
                        'bars_ago': len(recent_data) - recent_data.index.get_loc(idx) - 1
                    })
        
        return crosses
    
    def calculate_mean_reversion_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate mean reversion indicators"""
        df = data.copy()
        
        # RSI using pandas-ta
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Bollinger Bands using pandas-ta
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            df['BB_upper'] = bb['BBU_20_2.0']
            df['BB_middle'] = bb['BBM_20_2.0'] 
            df['BB_lower'] = bb['BBL_20_2.0']
        else:
            # Fallback calculation
            df['BB_middle'] = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            df['BB_upper'] = df['BB_middle'] + (std * 2)
            df['BB_lower'] = df['BB_middle'] - (std * 2)
        
        # Current values
        latest = df.iloc[-1]
        
        # RSI analysis
        rsi_current = latest['RSI'] if not pd.isna(latest['RSI']) else 50
        rsi_signal = 'neutral'
        if rsi_current < 30:
            rsi_signal = 'oversold'
        elif rsi_current > 70:
            rsi_signal = 'overbought'
        
        # Bollinger Bands analysis
        if not pd.isna(latest['BB_upper']) and not pd.isna(latest['BB_lower']):
            bb_range = latest['BB_upper'] - latest['BB_lower']
            if bb_range > 0:
                bb_position = (latest['Close'] - latest['BB_lower']) / bb_range
            else:
                bb_position = 0.5
        else:
            bb_position = 0.5
        
        bb_signal = 'neutral'
        if bb_position < 0.2:
            bb_signal = 'oversold'
        elif bb_position > 0.8:
            bb_signal = 'overbought'
        
        # Distance from 21 EMA (mean reversion level)
        ema_21 = latest.get('EMA_21', latest['Close'])
        if pd.isna(ema_21):
            ema_21 = latest['Close']
        distance_from_mean = ((latest['Close'] - ema_21) / ema_21) * 100 if ema_21 != 0 else 0
        
        return {
            'rsi_value': float(rsi_current),
            'rsi_signal': rsi_signal,
            'bb_position': float(bb_position),
            'bb_signal': bb_signal,
            'distance_from_mean_pct': float(distance_from_mean),
            'close_price': float(latest['Close']),
            'bb_upper': float(latest.get('BB_upper', latest['Close'])),
            'bb_lower': float(latest.get('BB_lower', latest['Close']))
        }
    
    def assess_market_choppiness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess if market is choppy/ranging"""
        df = data.copy()
        
        # ADX using pandas-ta
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx_data is not None and 'ADX_14' in adx_data.columns:
            df['ADX'] = adx_data['ADX_14']
        else:
            # Fallback: simple trend strength calculation
            df['ADX'] = df['Close'].rolling(14).apply(
                lambda x: abs(x.iloc[-1] - x.iloc[0]) / x.std() * 10 if x.std() > 0 else 0
            )
        
        # ATR using pandas-ta
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        if df['ATR'].isna().all():
            # Fallback ATR calculation
            df['TR'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            df['ATR'] = df['TR'].rolling(14).mean()
        
        # Price efficiency ratio
        lookback = min(20, len(df))
        if lookback > 1:
            price_change = abs(df['Close'].iloc[-1] - df['Close'].iloc[-lookback])
            total_movement = df['High'].tail(lookback).max() - df['Low'].tail(lookback).min()
            efficiency_ratio = price_change / total_movement if total_movement > 0 else 0
        else:
            efficiency_ratio = 0
        
        latest = df.iloc[-1]
        adx_value = latest['ADX'] if not pd.isna(latest['ADX']) else 25
        atr_value = latest['ATR'] if not pd.isna(latest['ATR']) else 0
        
        # Assess choppiness
        is_choppy = False
        choppiness_score = 0
        
        if adx_value < 25:  # Weak trend
            choppiness_score += 30
        if efficiency_ratio < 0.3:  # Low efficiency
            choppiness_score += 40
        
        # Recent volatility vs historical
        recent_atr = df['ATR'].tail(5).mean()
        historical_atr = df['ATR'].mean()
        
        if not pd.isna(recent_atr) and not pd.isna(historical_atr) and historical_atr > 0:
            if recent_atr > historical_atr * 1.5:
                choppiness_score += 30
        
        is_choppy = choppiness_score > 50
        
        return {
            'adx_value': float(adx_value),
            'atr_value': float(atr_value),
            'efficiency_ratio': float(efficiency_ratio),
            'is_choppy': is_choppy,
            'choppiness_score': choppiness_score,
            'trend_strength': 'weak' if adx_value < 25 else 'moderate' if adx_value < 40 else 'strong'
        }
    
    def analyze_trend(self, data: pd.DataFrame, timeframe: str = "unknown", symbol: str = "") -> Dict[str, Any]:
        """Comprehensive trend analysis"""
        
        # Calculate EMAs
        df_with_emas = self.calculate_emas(data)
        
        # Calculate slopes
        slopes = self.calculate_slopes(df_with_emas)
        
        # Analyze EMA alignment
        alignment = self.analyze_ema_alignment(df_with_emas)
        
        # Detect EMA crosses
        crosses = self.detect_ema_crosses(df_with_emas)
        
        # Mean reversion analysis
        mean_reversion = self.calculate_mean_reversion_signals(df_with_emas)
        
        # Market choppiness
        choppiness = self.assess_market_choppiness(df_with_emas)
        
        # Overall trend determination
        trend_direction = self.determine_trend_direction(alignment, slopes, crosses)
        
        return {
            'timeframe': timeframe,
            'symbol': symbol,
            'data_points': len(data),
            'analysis_time': datetime.now(),
            'slopes': slopes,
            'ema_alignment': alignment,
            'recent_crosses': crosses[-3:] if crosses else [],  # Last 3 crosses
            'mean_reversion': mean_reversion,
            'choppiness': choppiness,
            'trend_direction': trend_direction,
            'latest_price': float(data['Close'].iloc[-1]),
            'price_change_pct': float(((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100)
        }
    
    def determine_trend_direction(self, alignment: Dict, slopes: Dict, crosses: List) -> str:
        """Determine overall trend direction"""
        
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA alignment
        if alignment['alignment'] in ['strong_bullish', 'bullish']:
            bullish_signals += 2 if 'strong' in alignment['alignment'] else 1
        elif alignment['alignment'] in ['strong_bearish', 'bearish']:
            bearish_signals += 2 if 'strong' in alignment['alignment'] else 1
        
        # Slopes
        positive_slopes = sum(1 for slope in slopes.values() if slope > 0.05)  # 0.05% minimum
        negative_slopes = sum(1 for slope in slopes.values() if slope < -0.05)
        
        if positive_slopes > negative_slopes:
            bullish_signals += 1
        elif negative_slopes > positive_slopes:
            bearish_signals += 1
        
        # Recent crosses
        recent_bullish_crosses = sum(1 for cross in crosses if cross['type'] == 'bullish' and cross['bars_ago'] < 10)
        recent_bearish_crosses = sum(1 for cross in crosses if cross['type'] == 'bearish' and cross['bars_ago'] < 10)
        
        bullish_signals += recent_bullish_crosses
        bearish_signals += recent_bearish_crosses
        
        # Determine direction
        if bullish_signals > bearish_signals + 1:
            return 'bullish'
        elif bearish_signals > bullish_signals + 1:
            return 'bearish'
        else:
            return 'neutral'
    
    def generate_comprehensive_report(self, symbol: str, long_term: Dict, short_term: Dict, 
                                    long_term_data: pd.DataFrame, short_term_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive trading analysis report"""
        
        # Calculate trading score (0-100)
        trading_score = self.calculate_trading_score(long_term, short_term)
        
        # Market state assessment
        market_state = self.assess_market_state(long_term, short_term)
        
        # Risk assessment
        risk_assessment = self.assess_risk_level(long_term, short_term)
        
        # Trading opportunities
        opportunities = self.identify_trading_opportunities(long_term, short_term)
        
        # Generate detailed commentary
        commentary = self.generate_market_commentary(symbol, long_term, short_term, trading_score)
        
        return {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'trading_score': trading_score,
            'market_state': market_state,
            'risk_assessment': risk_assessment,
            'trend_direction': long_term['trend_direction'],
            'short_term_trend': short_term['trend_direction'],
            'long_term_analysis': long_term,
            'short_term_analysis': short_term,
            'opportunities': opportunities,
            'commentary': commentary,
            'latest_price': float(short_term_data['Close'].iloc[-1]),
            'price_change_24h': short_term['price_change_pct'],
            'is_choppy_market': long_term['choppiness']['is_choppy'],
            'trend_strength': long_term['choppiness']['trend_strength']
        }
    
    def calculate_trading_score(self, long_term: Dict, short_term: Dict) -> int:
        """Calculate trading opportunity score (0-100)"""
        score = 50  # Base score
        
        # Long-term trend clarity (+/- 20 points)
        lt_alignment = long_term['ema_alignment']['strength']
        if lt_alignment > 80:
            score += 20
        elif lt_alignment > 60:
            score += 10
        elif lt_alignment < 40:
            score -= 15
        
        # Trend agreement between timeframes (+/- 15 points)
        if long_term['trend_direction'] == short_term['trend_direction'] and long_term['trend_direction'] != 'neutral':
            score += 15
        elif long_term['trend_direction'] != short_term['trend_direction']:
            score -= 10
        
        # Market choppiness penalty (-30 points max)
        if long_term['choppiness']['is_choppy']:
            score -= 30
        elif long_term['choppiness']['choppiness_score'] > 30:
            score -= 15
        
        # Mean reversion opportunity (+/- 10 points)
        mr = short_term['mean_reversion']
        if mr['rsi_signal'] in ['oversold', 'overbought'] and mr['bb_signal'] in ['oversold', 'overbought']:
            score += 10
        
        # Recent crosses (+/- 10 points)
        recent_crosses = short_term['recent_crosses']
        if recent_crosses:
            latest_cross = recent_crosses[-1]
            if latest_cross['bars_ago'] < 5:  # Very recent cross
                score += 10 if latest_cross['type'] == long_term['trend_direction'] else -5
        
        # Trend strength bonus/penalty
        adx = long_term['choppiness']['adx_value']
        if adx > 40:
            score += 10  # Strong trend
        elif adx < 20:
            score -= 10  # Very weak trend
        
        return max(0, min(100, score))
    
    def assess_market_state(self, long_term: Dict, short_term: Dict) -> str:
        """Assess current market state"""
        
        is_choppy = long_term['choppiness']['is_choppy']
        trend_strength = long_term['choppiness']['trend_strength']
        lt_trend = long_term['trend_direction']
        st_trend = short_term['trend_direction']
        
        if is_choppy:
            return "Choppy/Ranging Market - Avoid"
        
        if trend_strength == 'strong':
            if lt_trend == st_trend and lt_trend != 'neutral':
                return f"Strong {lt_trend.title()} Trend - High Confidence"
            else:
                return "Strong Trend with Mixed Signals"
        
        elif trend_strength == 'moderate':
            if lt_trend == st_trend and lt_trend != 'neutral':
                return f"Moderate {lt_trend.title()} Trend - Good Opportunity"
            else:
                return "Moderate Trend - Wait for Confirmation"
        
        else:  # weak trend
            return "Weak Trend - Low Conviction"
    
    def assess_risk_level(self, long_term: Dict, short_term: Dict) -> str:
        """Assess risk level for trading"""
        
        risk_factors = 0
        
        # Choppiness
        if long_term['choppiness']['is_choppy']:
            risk_factors += 3
        elif long_term['choppiness']['choppiness_score'] > 30:
            risk_factors += 1
        
        # Conflicting timeframes
        if long_term['trend_direction'] != short_term['trend_direction']:
            risk_factors += 2
        
        # Weak trend
        if long_term['choppiness']['adx_value'] < 20:
            risk_factors += 2
        
        # Mixed EMA alignment
        if long_term['ema_alignment']['alignment'] == 'mixed':
            risk_factors += 1
        
        if risk_factors >= 5:
            return "High Risk"
        elif risk_factors >= 3:
            return "Moderate Risk"
        elif risk_factors >= 1:
            return "Low-Moderate Risk"
        else:
            return "Low Risk"
    
    def identify_trading_opportunities(self, long_term: Dict, short_term: Dict) -> List[str]:
        """Identify specific trading opportunities"""
        opportunities = []
        
        # Trend following opportunities
        if (long_term['trend_direction'] == short_term['trend_direction'] and 
            long_term['trend_direction'] != 'neutral' and
            not long_term['choppiness']['is_choppy']):
            
            opportunities.append(f"Trend Following: {long_term['trend_direction'].title()} direction")
        
        # Mean reversion opportunities
        mr = short_term['mean_reversion']
        if mr['rsi_signal'] in ['oversold', 'overbought'] and not long_term['choppiness']['is_choppy']:
            opposite_direction = 'bullish' if mr['rsi_signal'] == 'oversold' else 'bearish'
            opportunities.append(f"Mean Reversion: {opposite_direction.title()} bounce expected")
        
        # Breakout opportunities
        recent_crosses = short_term['recent_crosses']
        if recent_crosses:
            latest_cross = recent_crosses[-1]
            if latest_cross['bars_ago'] < 3:
                opportunities.append(f"EMA Breakout: {latest_cross['type'].title()} momentum")
        
        # Retracement opportunities
        if (long_term['trend_direction'] != 'neutral' and 
            short_term['trend_direction'] != long_term['trend_direction'] and
            not long_term['choppiness']['is_choppy']):
            opportunities.append(f"Retracement Buy: {long_term['trend_direction'].title()} trend continuation")
        
        if not opportunities:
            opportunities.append("No clear opportunities - Wait for better setup")
        
        return opportunities
    
    def generate_market_commentary(self, symbol: str, long_term: Dict, short_term: Dict, trading_score: int) -> str:
        """Generate detailed market commentary"""
        
        commentary_parts = []
        
        # Opening statement
        commentary_parts.append(f"Analysis for {symbol} (Score: {trading_score}/100)")
        commentary_parts.append("-" * 50)
        
        # Long-term trend
        lt_trend = long_term['trend_direction']
        lt_strength = long_term['choppiness']['trend_strength']
        commentary_parts.append(f"Long-term Trend: {lt_trend.title()} ({lt_strength} strength)")
        
        # Short-term trend
        st_trend = short_term['trend_direction']
        commentary_parts.append(f"Short-term Trend: {st_trend.title()}")
        
        # EMA alignment
        alignment = long_term['ema_alignment']
        commentary_parts.append(f"EMA Alignment: {alignment['alignment'].replace('_', ' ').title()} ({alignment['strength']:.1f}%)")
        
        # Market condition
        if long_term['choppiness']['is_choppy']:
            commentary_parts.append("⚠️  WARNING: Market appears choppy - avoid trading")
        else:
            commentary_parts.append("✅ Market conditions suitable for trading")
        
        # Mean reversion status
        mr = short_term['mean_reversion']
        commentary_parts.append(f"RSI: {mr['rsi_value']:.1f} ({mr['rsi_signal']})")
        commentary_parts.append(f"Bollinger Band Position: {mr['bb_position']:.2f} ({mr['bb_signal']})")
        
        # Recent activity
        if short_term['recent_crosses']:
            latest_cross = short_term['recent_crosses'][-1]
            commentary_parts.append(f"Recent Cross: {latest_cross['type'].title()} EMA{latest_cross['fast_ema']}/EMA{latest_cross['slow_ema']} ({latest_cross['bars_ago']} bars ago)")
        
        # Trading recommendation
        if trading_score >= 75:
            commentary_parts.append("🚀 STRONG BUY SIGNAL - High probability setup")
        elif trading_score >= 60:
            commentary_parts.append("📈 GOOD OPPORTUNITY - Favorable conditions")
        elif trading_score >= 40:
            commentary_parts.append("⚖️  NEUTRAL - Wait for better confirmation")
        else:
            commentary_parts.append("❌ AVOID - Poor trading conditions")
        
        return "\n".join(commentary_parts)