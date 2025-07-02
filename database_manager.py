"""
Database Manager Module
MongoDB integration for storing analysis results, parameters, and scheduling
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import pandas as pd
from dataclasses import asdict

class DatabaseManager:
    def __init__(self, connection_string: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # MongoDB connection string (Atlas free tier)
        self.connection_string = connection_string or os.getenv(
            'MONGODB_CONNECTION_STRING',
            'mongodb://localhost:27017/'  # Fallback to local
        )
        
        self.client = None
        self.db = None
        self.collections = {}
        
        # Collection names
        self.COLLECTIONS = {
            'analysis_results': 'analysis_results',
            'optimal_parameters': 'optimal_parameters',
            'market_data': 'market_data',
            'scheduled_jobs': 'scheduled_jobs',
            'performance_tracking': 'performance_tracking'
        }
        
        self._connect()
        self._setup_collections()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            
            # Use database
            self.db = self.client['trend_filter_pro']
            self.logger.info("Successfully connected to MongoDB")
            
        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def _setup_collections(self):
        """Setup collections with indexes"""
        try:
            for collection_name in self.COLLECTIONS.values():
                self.collections[collection_name] = self.db[collection_name]
            
            # Create indexes for better performance
            self._create_indexes()
            self.logger.info("Collections and indexes setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up collections: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for efficient querying"""
        
        # Analysis results indexes
        self.collections['analysis_results'].create_index([
            ('symbol', ASCENDING),
            ('timestamp', DESCENDING)
        ])
        self.collections['analysis_results'].create_index([('timestamp', DESCENDING)])
        
        # Optimal parameters indexes
        self.collections['optimal_parameters'].create_index([
            ('symbol', ASCENDING),
            ('timeframe', ASCENDING),
            ('optimization_date', DESCENDING)
        ])
        
        # Market data indexes
        self.collections['market_data'].create_index([
            ('symbol', ASCENDING),
            ('timestamp', ASCENDING)
        ], unique=True)  # Prevent duplicates
        
        # Scheduled jobs indexes
        self.collections['scheduled_jobs'].create_index([
            ('next_run', ASCENDING),
            ('active', ASCENDING)
        ])
        
        # Performance tracking indexes
        self.collections['performance_tracking'].create_index([
            ('symbol', ASCENDING),
            ('period_start', DESCENDING)
        ])
    
    def store_analysis_result(self, analysis_result: Dict[str, Any]) -> str:
        """Store analysis result in database"""
        try:
            # Add metadata
            document = {
                'symbol': analysis_result['symbol'],
                'timestamp': datetime.now(),
                'analysis_timestamp': analysis_result['analysis_timestamp'],
                'trading_score': analysis_result['trading_score'],
                'market_state': analysis_result['market_state'],
                'risk_assessment': analysis_result['risk_assessment'],
                'trend_direction': analysis_result['trend_direction'],
                'short_term_trend': analysis_result['short_term_trend'],
                'latest_price': analysis_result['latest_price'],
                'price_change_24h': analysis_result['price_change_24h'],
                'is_choppy_market': analysis_result['is_choppy_market'],
                'trend_strength': analysis_result['trend_strength'],
                'opportunities': analysis_result['opportunities'],
                'commentary': analysis_result['commentary'],
                'long_term_analysis': analysis_result['long_term_analysis'],
                'short_term_analysis': analysis_result['short_term_analysis']
            }
            
            result = self.collections['analysis_results'].insert_one(document)
            self.logger.info(f"Stored analysis result for {analysis_result['symbol']}")
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Error storing analysis result: {str(e)}")
            raise
    
    def store_optimal_parameters(self, optimization_result, symbol: str, timeframe: str) -> str:
        """Store optimal parameters from optimization"""
        try:
            document = {
                'symbol': symbol,
                'timeframe': timeframe,
                'optimization_date': datetime.now(),
                'parameters': optimization_result.best_parameters.to_dict(),
                'performance_metrics': optimization_result.performance_metrics,
                'validation_score': optimization_result.validation_score,
                'sample_size': optimization_result.sample_size,
                'overfitting_score': optimization_result.overfitting_score,
                'confidence_level': optimization_result.confidence_level,
                'optimization_period': optimization_result.optimization_period
            }
            
            # Update existing or insert new
            filter_query = {
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            result = self.collections['optimal_parameters'].replace_one(
                filter_query, document, upsert=True
            )
            
            self.logger.info(f"Stored optimal parameters for {symbol} ({timeframe})")
            return str(result.upserted_id) if result.upserted_id else "updated"
            
        except Exception as e:
            self.logger.error(f"Error storing optimal parameters: {str(e)}")
            raise
    
    def get_latest_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest analysis for a symbol"""
        try:
            result = self.collections['analysis_results'].find_one(
                {'symbol': symbol},
                sort=[('timestamp', DESCENDING)]
            )
            
            if result:
                # Convert ObjectId to string
                result['_id'] = str(result['_id'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting latest analysis: {str(e)}")
            return None
    
    def get_optimal_parameters(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get optimal parameters for symbol and timeframe"""
        try:
            result = self.collections['optimal_parameters'].find_one(
                {'symbol': symbol, 'timeframe': timeframe},
                sort=[('optimization_date', DESCENDING)]
            )
            
            if result:
                result['_id'] = str(result['_id'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting optimal parameters: {str(e)}")
            return None
    
    def store_market_data_point(self, symbol: str, timestamp: datetime, 
                               open_price: float, high: float, low: float, 
                               close: float, volume: int) -> bool:
        """Store individual market data point (prevents duplicates)"""
        try:
            document = {
                'symbol': symbol,
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'created_at': datetime.now()
            }
            
            self.collections['market_data'].insert_one(document)
            return True
            
        except DuplicateKeyError:
            # Data point already exists, skip
            return False
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")
            return False
    
    def get_latest_data_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get the timestamp of the latest data point for a symbol"""
        try:
            result = self.collections['market_data'].find_one(
                {'symbol': symbol},
                sort=[('timestamp', DESCENDING)],
                projection={'timestamp': 1}
            )
            
            return result['timestamp'] if result else None
            
        except Exception as e:
            self.logger.error(f"Error getting latest data timestamp: {str(e)}")
            return None
    
    def bulk_store_market_data(self, symbol: str, data: pd.DataFrame) -> int:
        """Bulk store market data, skipping duplicates"""
        try:
            documents = []
            
            for timestamp, row in data.iterrows():
                document = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'created_at': datetime.now()
                }
                documents.append(document)
            
            # Use ordered=False to continue on duplicates
            result = self.collections['market_data'].insert_many(
                documents, ordered=False
            )
            
            self.logger.info(f"Stored {len(result.inserted_ids)} new data points for {symbol}")
            return len(result.inserted_ids)
            
        except Exception as e:
            # Count successful inserts even if some failed due to duplicates
            error_msg = str(e)
            if "duplicate key error" in error_msg:
                # Extract number of successful inserts from error message
                return len([d for d in documents]) - error_msg.count("duplicate")
            else:
                self.logger.error(f"Error bulk storing market data: {str(e)}")
                return 0
    
    def schedule_job(self, symbol: str, job_type: str, interval_hours: int, 
                     parameters: Dict[str, Any] = None) -> str:
        """Schedule a recurring job"""
        try:
            next_run = datetime.now() + timedelta(hours=interval_hours)
            
            document = {
                'symbol': symbol,
                'job_type': job_type,  # 'analysis', 'optimization', 'data_update'
                'interval_hours': interval_hours,
                'next_run': next_run,
                'last_run': None,
                'parameters': parameters or {},
                'active': True,
                'created_at': datetime.now()
            }
            
            result = self.collections['scheduled_jobs'].insert_one(document)
            self.logger.info(f"Scheduled {job_type} job for {symbol}")
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Error scheduling job: {str(e)}")
            raise
    
    def get_due_jobs(self) -> List[Dict[str, Any]]:
        """Get jobs that are due to run"""
        try:
            current_time = datetime.now()
            
            jobs = list(self.collections['scheduled_jobs'].find({
                'next_run': {'$lte': current_time},
                'active': True
            }))
            
            # Convert ObjectIds to strings
            for job in jobs:
                job['_id'] = str(job['_id'])
            
            return jobs
            
        except Exception as e:
            self.logger.error(f"Error getting due jobs: {str(e)}")
            return []
    
    def update_job_run(self, job_id: str, next_run: datetime, success: bool = True):
        """Update job after execution"""
        try:
            from bson import ObjectId
            
            update_data = {
                'last_run': datetime.now(),
                'next_run': next_run,
                'last_success': success
            }
            
            self.collections['scheduled_jobs'].update_one(
                {'_id': ObjectId(job_id)},
                {'$set': update_data}
            )
            
        except Exception as e:
            self.logger.error(f"Error updating job run: {str(e)}")
    
    def get_performance_summary(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for a symbol over the last N days"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Get recent analyses
            analyses = list(self.collections['analysis_results'].find({
                'symbol': symbol,
                'timestamp': {'$gte': start_date}
            }).sort('timestamp', DESCENDING))
            
            if not analyses:
                return {'symbol': symbol, 'analyses_count': 0}
            
            # Calculate statistics
            scores = [a['trading_score'] for a in analyses]
            
            summary = {
                'symbol': symbol,
                'period_days': days,
                'analyses_count': len(analyses),
                'avg_trading_score': sum(scores) / len(scores),
                'max_trading_score': max(scores),
                'min_trading_score': min(scores),
                'latest_score': scores[0] if scores else 0,
                'trend_direction_counts': {},
                'market_state_counts': {},
                'last_updated': analyses[0]['timestamp'] if analyses else None
            }
            
            # Count trend directions
            trends = [a['trend_direction'] for a in analyses]
            for trend in set(trends):
                summary['trend_direction_counts'][trend] = trends.count(trend)
            
            # Count market states
            states = [a['market_state'] for a in analyses]
            for state in set(states):
                summary['market_state_counts'][state] = states.count(state)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_best_parameters_by_period(self, period_days: int = 7) -> List[Dict[str, Any]]:
        """Get best performing parameters for each symbol over a period"""
        try:
            start_date = datetime.now() - timedelta(days=period_days)
            
            # Aggregate best parameters by symbol
            pipeline = [
                {
                    '$match': {
                        'optimization_date': {'$gte': start_date}
                    }
                },
                {
                    '$sort': {
                        'symbol': 1,
                        'validation_score': -1
                    }
                },
                {
                    '$group': {
                        '_id': '$symbol',
                        'best_params': {'$first': '$$ROOT'}
                    }
                }
            ]
            
            results = list(self.collections['optimal_parameters'].aggregate(pipeline))
            
            # Format results
            formatted_results = []
            for result in results:
                best_params = result['best_params']
                best_params['_id'] = str(best_params['_id'])
                formatted_results.append({
                    'symbol': result['_id'],
                    'period_days': period_days,
                    'best_parameters': best_params
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error getting best parameters: {str(e)}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage storage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean old analysis results
            result1 = self.collections['analysis_results'].delete_many({
                'timestamp': {'$lt': cutoff_date}
            })
            
            # Clean old market data (keep more for historical analysis)
            market_cutoff = datetime.now() - timedelta(days=days_to_keep * 2)
            result2 = self.collections['market_data'].delete_many({
                'created_at': {'$lt': market_cutoff}
            })
            
            self.logger.info(f"Cleaned up {result1.deleted_count} old analysis results")
            self.logger.info(f"Cleaned up {result2.deleted_count} old market data points")
            
            return {
                'analysis_results_deleted': result1.deleted_count,
                'market_data_deleted': result2.deleted_count
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
            return {'error': str(e)}
    
    def close_connection(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.logger.info("Database connection closed")