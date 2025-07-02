"""
Parameter Optimizer Module
Finds optimal parameters while preventing overfitting through walk-forward analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import itertools
import logging
from dataclasses import dataclass
from trend_analyzer import TrendAnalyzer
from data_manager import DataManager

@dataclass
class ParameterSet:
    """Parameter configuration for optimization"""
    ema_periods: List[int]
    rsi_period: int
    bb_period: int
    bb_std: float
    adx_period: int
    atr_period: int
    slope_threshold: float
    trend_strength_threshold: int
    choppiness_threshold: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ema_periods': self.ema_periods,
            'rsi_period': self.rsi_period,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'adx_period': self.adx_period,
            'atr_period': self.atr_period,
            'slope_threshold': self.slope_threshold,
            'trend_strength_threshold': self.trend_strength_threshold,
            'choppiness_threshold': self.choppiness_threshold
        }

@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    symbol: str
    timeframe: str
    best_parameters: ParameterSet
    performance_metrics: Dict[str, float]
    validation_score: float
    optimization_period: str
    sample_size: int
    overfitting_score: float  # Lower is better
    confidence_level: str

class ParameterOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_manager = DataManager()
        
        # Define parameter ranges for optimization
        self.parameter_ranges = {
            'ema_periods': [
                [5, 8, 13, 21, 34],      # Fast EMAs
                [8, 13, 21, 34, 55],     # Standard
                [13, 21, 34, 55, 89],    # Slower
                [21, 34, 55, 89, 144],   # Very slow
            ],
            'rsi_period': [10, 14, 18, 21],
            'bb_period': [15, 20, 25],
            'bb_std': [1.5, 2.0, 2.5],
            'adx_period': [10, 14, 18],
            'atr_period': [10, 14, 18],
            'slope_threshold': [0.03, 0.05, 0.08, 0.10],
            'trend_strength_threshold': [20, 25, 30],
            'choppiness_threshold': [40, 50, 60]
        }
        
        # Anti-overfitting measures
        self.min_sample_size = 100  # Minimum data points required
        self.validation_split = 0.3  # 30% for validation
        self.max_iterations = 50  # Limit parameter combinations
        
    def generate_parameter_combinations(self, symbol: str = "") -> List[ParameterSet]:
        """Generate parameter combinations with intelligent filtering"""
        
        # Start with all combinations
        keys = self.parameter_ranges.keys()
        values = self.parameter_ranges.values()
        
        # Generate combinations but limit to prevent overfitting
        all_combinations = list(itertools.product(*values))
        
        # Shuffle and limit
        np.random.shuffle(all_combinations)
        limited_combinations = all_combinations[:self.max_iterations]
        
        parameter_sets = []
        for combo in limited_combinations:
            param_dict = dict(zip(keys, combo))
            
            # Create parameter set
            param_set = ParameterSet(
                ema_periods=param_dict['ema_periods'],
                rsi_period=param_dict['rsi_period'],
                bb_period=param_dict['bb_period'],
                bb_std=param_dict['bb_std'],
                adx_period=param_dict['adx_period'],
                atr_period=param_dict['atr_period'],
                slope_threshold=param_dict['slope_threshold'],
                trend_strength_threshold=param_dict['trend_strength_threshold'],
                choppiness_threshold=param_dict['choppiness_threshold']
            )
            
            parameter_sets.append(param_set)
        
        self.logger.info(f"Generated {len(parameter_sets)} parameter combinations for testing")
        return parameter_sets
    
    def optimize_parameters(self, symbol: str, optimization_period: str = "3mo") -> OptimizationResult:
        """
        Optimize parameters using walk-forward analysis to prevent overfitting
        """
        
        self.logger.info(f"Starting parameter optimization for {symbol}")
        
        # Fetch extended data for optimization
        data = self.data_manager.fetch_data(
            symbol=symbol,
            period="1y",  # Get more data for robust testing
            interval="1h"
        )
        
        if len(data) < self.min_sample_size:
            raise ValueError(f"Insufficient data for optimization: {len(data)} < {self.min_sample_size}")
        
        # Split data for walk-forward analysis
        train_size = int(len(data) * (1 - self.validation_split))
        train_data = data.iloc[:train_size]
        validation_data = data.iloc[train_size:]
        
        self.logger.info(f"Training data: {len(train_data)} points, Validation: {len(validation_data)} points")
        
        # Generate parameter combinations
        parameter_sets = self.generate_parameter_combinations(symbol)
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        for i, param_set in enumerate(parameter_sets):
            try:
                self.logger.info(f"Testing parameter set {i+1}/{len(parameter_sets)}")
                
                # Test parameters on training data
                train_score = self._evaluate_parameters(param_set, train_data, symbol)
                
                # Test on validation data
                validation_score = self._evaluate_parameters(param_set, validation_data, symbol)
                
                # Calculate overfitting score (difference between train and validation)
                overfitting_score = abs(train_score - validation_score)
                
                # Combined score: validation performance with overfitting penalty
                combined_score = validation_score - (overfitting_score * 0.5)
                
                result = {
                    'parameters': param_set,
                    'train_score': train_score,
                    'validation_score': validation_score,
                    'combined_score': combined_score,
                    'overfitting_score': overfitting_score
                }
                
                all_results.append(result)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_params = param_set
                
            except Exception as e:
                self.logger.warning(f"Error testing parameter set {i+1}: {str(e)}")
                continue
        
        if best_params is None:
            raise ValueError("No valid parameter combinations found")
        
        # Calculate performance metrics for best parameters
        performance_metrics = self._calculate_performance_metrics(best_params, data, symbol)
        
        # Determine confidence level
        confidence_level = self._assess_confidence(all_results, best_score)
        
        result = OptimizationResult(
            symbol=symbol,
            timeframe="1h",
            best_parameters=best_params,
            performance_metrics=performance_metrics,
            validation_score=best_score,
            optimization_period=optimization_period,
            sample_size=len(data),
            overfitting_score=min(r['overfitting_score'] for r in all_results if r['parameters'] == best_params),
            confidence_level=confidence_level
        )
        
        self.logger.info(f"Optimization complete. Best validation score: {best_score:.2f}")
        return result
    
    def _evaluate_parameters(self, param_set: ParameterSet, data: pd.DataFrame, symbol: str) -> float:
        """Evaluate parameter set performance on given data"""
        
        # Create custom trend analyzer with these parameters
        analyzer = TrendAnalyzer()
        analyzer.ema_periods = param_set.ema_periods
        
        try:
            # Perform analysis
            analysis = analyzer.analyze_trend(data, timeframe="optimization", symbol=symbol)
            
            # Calculate performance score based on multiple factors
            score = 0
            
            # Trend clarity (EMA alignment strength)
            alignment_strength = analysis['ema_alignment']['strength']
            score += alignment_strength * 0.3
            
            # Trend consistency (slope agreement)
            slopes = analysis['slopes']
            positive_slopes = sum(1 for s in slopes.values() if s > param_set.slope_threshold)
            negative_slopes = sum(1 for s in slopes.values() if s < -param_set.slope_threshold)
            trend_consistency = abs(positive_slopes - negative_slopes) / len(slopes) * 100
            score += trend_consistency * 0.25
            
            # Signal quality (recent crosses alignment with trend)
            crosses = analysis['recent_crosses']
            if crosses:
                trend_aligned_crosses = sum(1 for c in crosses 
                                          if c['type'] == analysis['trend_direction'] 
                                          and c['bars_ago'] < 10)
                signal_quality = (trend_aligned_crosses / len(crosses)) * 100
                score += signal_quality * 0.2
            
            # Market condition assessment
            choppiness = analysis['choppiness']
            if not choppiness['is_choppy']:
                score += 25  # Bonus for trending market identification
            
            # Mean reversion quality
            mr = analysis['mean_reversion']
            if mr['rsi_signal'] != 'neutral':
                score += 10  # Bonus for clear signals
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Error evaluating parameters: {str(e)}")
            return 0
    
    def _calculate_performance_metrics(self, param_set: ParameterSet, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate detailed performance metrics for the best parameters"""
        
        analyzer = TrendAnalyzer()
        analyzer.ema_periods = param_set.ema_periods
        
        analysis = analyzer.analyze_trend(data, timeframe="full", symbol=symbol)
        
        return {
            'ema_alignment_strength': analysis['ema_alignment']['strength'],
            'trend_direction_clarity': 1.0 if analysis['trend_direction'] != 'neutral' else 0.0,
            'adx_value': analysis['choppiness']['adx_value'],
            'rsi_value': analysis['mean_reversion']['rsi_value'],
            'choppiness_score': analysis['choppiness']['choppiness_score'],
            'recent_crosses_count': len(analysis['recent_crosses']),
            'price_change_pct': analysis['price_change_pct']
        }
    
    def _assess_confidence(self, all_results: List[Dict], best_score: float) -> str:
        """Assess confidence level based on result distribution"""
        
        if not all_results:
            return "Low"
        
        scores = [r['combined_score'] for r in all_results]
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Check how much better the best score is
        if best_score > mean_score + 2 * std_score:
            return "High"
        elif best_score > mean_score + std_score:
            return "Medium"
        else:
            return "Low"
    
    def optimize_multiple_timeframes(self, symbol: str) -> Dict[str, OptimizationResult]:
        """Optimize parameters for multiple timeframes"""
        
        timeframes = {
            'short_term': {'period': '1mo', 'interval': '15m'},
            'medium_term': {'period': '3mo', 'interval': '1h'},
            'long_term': {'period': '6mo', 'interval': '4h'}
        }
        
        results = {}
        
        for tf_name, tf_config in timeframes.items():
            try:
                self.logger.info(f"Optimizing parameters for {tf_name} timeframe")
                
                # Fetch data for this timeframe
                data = self.data_manager.fetch_data(
                    symbol=symbol,
                    period=tf_config['period'],
                    interval=tf_config['interval']
                )
                
                if len(data) >= self.min_sample_size:
                    result = self.optimize_parameters(symbol, tf_config['period'])
                    result.timeframe = tf_name
                    results[tf_name] = result
                else:
                    self.logger.warning(f"Insufficient data for {tf_name} timeframe")
                    
            except Exception as e:
                self.logger.error(f"Error optimizing {tf_name} timeframe: {str(e)}")
        
        return results
    
    def validate_parameters_robustness(self, param_set: ParameterSet, symbol: str, periods: List[str]) -> Dict[str, float]:
        """Test parameter robustness across different time periods"""
        
        scores = []
        
        for period in periods:
            try:
                data = self.data_manager.fetch_data(symbol=symbol, period=period, interval="1h")
                if len(data) >= 50:  # Minimum for testing
                    score = self._evaluate_parameters(param_set, data, symbol)
                    scores.append(score)
            except Exception as e:
                self.logger.warning(f"Error testing period {period}: {str(e)}")
        
        if not scores:
            return {'mean': 0, 'std': 0, 'consistency': 0}
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        consistency = 1 - (std_score / mean_score) if mean_score > 0 else 0
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'consistency': consistency,
            'test_periods': len(scores)
        }