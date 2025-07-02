"""
Job Scheduler Module
Handles automated execution of analysis, optimization, and data updates
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys

from database_manager import DatabaseManager
from trend_analyzer import TrendAnalyzer
from data_manager import DataManager
from parameter_optimizer import ParameterOptimizer
from utils import save_results_to_csv

class JobScheduler:
    def __init__(self, db_manager: DatabaseManager, max_workers: int = 3):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.data_manager = DataManager()
        self.trend_analyzer = TrendAnalyzer()
        self.parameter_optimizer = ParameterOptimizer()
        
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Job type handlers
        self.job_handlers = {
            'analysis': self._run_analysis_job,
            'optimization': self._run_optimization_job,
            'data_update': self._run_data_update_job,
            'cleanup': self._run_cleanup_job
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def start(self, check_interval: int = 60):
        """Start the job scheduler"""
        self.logger.info("Starting job scheduler...")
        self.running = True
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get due jobs
                due_jobs = self.db_manager.get_due_jobs()
                
                if due_jobs:
                    self.logger.info(f"Found {len(due_jobs)} due jobs")
                    self._execute_jobs(due_jobs)
                
                # Wait before next check
                self.shutdown_event.wait(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler main loop: {str(e)}")
                time.sleep(check_interval)
        
        self.logger.info("Job scheduler stopped")
    
    def stop(self):
        """Stop the job scheduler"""
        self.logger.info("Stopping job scheduler...")
        self.running = False
        self.shutdown_event.set()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    def _execute_jobs(self, jobs: List[Dict[str, Any]]):
        """Execute multiple jobs concurrently"""
        
        # Submit jobs to thread pool
        future_to_job = {
            self.executor.submit(self._execute_job, job): job 
            for job in jobs
        }
        
        # Process completed jobs
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                self.logger.info(f"Job {job['_id']} completed: {result}")
            except Exception as e:
                self.logger.error(f"Job {job['_id']} failed: {str(e)}")
    
    def _execute_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single job"""
        
        job_id = job['_id']
        job_type = job['job_type']
        symbol = job['symbol']
        
        self.logger.info(f"Executing {job_type} job for {symbol}")
        
        try:
            # Get job handler
            handler = self.job_handlers.get(job_type)
            if not handler:
                raise ValueError(f"Unknown job type: {job_type}")
            
            # Execute job
            result = handler(job)
            
            # Calculate next run time
            next_run = datetime.now() + timedelta(hours=job['interval_hours'])
            
            # Update job in database
            self.db_manager.update_job_run(job_id, next_run, success=True)
            
            return {
                'job_id': job_id,
                'status': 'success',
                'result': result,
                'next_run': next_run
            }
            
        except Exception as e:
            # Update job with failure
            next_run = datetime.now() + timedelta(hours=1)  # Retry in 1 hour
            self.db_manager.update_job_run(job_id, next_run, success=False)
            
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': str(e),
                'next_run': next_run
            }
    
    def _run_analysis_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run trend analysis job"""
        
        symbol = job['symbol']
        parameters = job.get('parameters', {})
        
        # Get optimal parameters if available
        optimal_params = self.db_manager.get_optimal_parameters(symbol, 'medium_term')
        if optimal_params:
            # Apply optimal parameters to analyzer
            self.trend_analyzer.ema_periods = optimal_params['parameters']['ema_periods']
        
        # Fetch data
        long_term_data = self.data_manager.fetch_data(
            symbol, 
            period=parameters.get('long_term_period', '3mo'),
            interval=parameters.get('long_term_interval', '1h')
        )
        
        short_term_data = self.data_manager.fetch_data(
            symbol,
            period=parameters.get('short_term_period', '5d'), 
            interval=parameters.get('short_term_interval', '5m')
        )
        
        if long_term_data.empty or short_term_data.empty:
            raise ValueError(f"Failed to fetch data for {symbol}")
        
        # Perform analysis
        long_term_analysis = self.trend_analyzer.analyze_trend(
            long_term_data, timeframe="long_term", symbol=symbol
        )
        
        short_term_analysis = self.trend_analyzer.analyze_trend(
            short_term_data, timeframe="short_term", symbol=symbol
        )
        
        # Generate report
        report = self.trend_analyzer.generate_comprehensive_report(
            symbol, long_term_analysis, short_term_analysis,
            long_term_data, short_term_data
        )
        
        # Store in database
        result_id = self.db_manager.store_analysis_result(report)
        
        # Also store market data points
        data_points_stored = self.db_manager.bulk_store_market_data(symbol, short_term_data)
        
        return {
            'analysis_result_id': result_id,
            'trading_score': report['trading_score'],
            'market_state': report['market_state'],
            'data_points_stored': data_points_stored
        }
    
    def _run_optimization_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run parameter optimization job"""
        
        symbol = job['symbol']
        parameters = job.get('parameters', {})
        
        # Run optimization for multiple timeframes
        optimization_results = self.parameter_optimizer.optimize_multiple_timeframes(symbol)
        
        stored_results = {}
        for timeframe, result in optimization_results.items():
            result_id = self.db_manager.store_optimal_parameters(result, symbol, timeframe)
            stored_results[timeframe] = {
                'result_id': result_id,
                'validation_score': result.validation_score,
                'confidence_level': result.confidence_level
            }
        
        return {
            'optimized_timeframes': list(optimization_results.keys()),
            'results': stored_results
        }
    
    def _run_data_update_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run data update job to fetch latest market data"""
        
        symbol = job['symbol']
        parameters = job.get('parameters', {})
        
        # Get latest data timestamp from database
        latest_timestamp = self.db_manager.get_latest_data_timestamp(symbol)
        
        # Determine how much data to fetch
        if latest_timestamp:
            # Fetch data since last update
            days_since = (datetime.now() - latest_timestamp).days
            period = f"{min(days_since + 1, 30)}d"  # Max 30 days
        else:
            # First time, fetch more data
            period = parameters.get('initial_period', '1mo')
        
        # Fetch different intervals
        intervals = ['1h', '5m', '15m']
        total_stored = 0
        
        for interval in intervals:
            try:
                data = self.data_manager.fetch_data(symbol, period=period, interval=interval)
                if not data.empty:
                    stored_count = self.db_manager.bulk_store_market_data(symbol, data)
                    total_stored += stored_count
                    self.logger.info(f"Stored {stored_count} {interval} data points for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to update {interval} data for {symbol}: {str(e)}")
        
        return {
            'symbol': symbol,
            'total_data_points_stored': total_stored,
            'latest_timestamp': latest_timestamp
        }
    
    def _run_cleanup_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run database cleanup job"""
        
        parameters = job.get('parameters', {})
        days_to_keep = parameters.get('days_to_keep', 90)
        
        cleanup_result = self.db_manager.cleanup_old_data(days_to_keep)
        
        return cleanup_result
    
    def schedule_analysis_jobs(self, symbols: List[str], interval_hours: int = 4):
        """Schedule regular analysis jobs for multiple symbols"""
        
        for symbol in symbols:
            try:
                job_id = self.db_manager.schedule_job(
                    symbol=symbol,
                    job_type='analysis',
                    interval_hours=interval_hours,
                    parameters={
                        'long_term_period': '3mo',
                        'long_term_interval': '1h',
                        'short_term_period': '5d',
                        'short_term_interval': '5m'
                    }
                )
                self.logger.info(f"Scheduled analysis job for {symbol}: {job_id}")
            except Exception as e:
                self.logger.error(f"Failed to schedule analysis job for {symbol}: {str(e)}")
    
    def schedule_optimization_jobs(self, symbols: List[str], interval_hours: int = 168):  # Weekly
        """Schedule parameter optimization jobs"""
        
        for symbol in symbols:
            try:
                job_id = self.db_manager.schedule_job(
                    symbol=symbol,
                    job_type='optimization',
                    interval_hours=interval_hours,
                    parameters={}
                )
                self.logger.info(f"Scheduled optimization job for {symbol}: {job_id}")
            except Exception as e:
                self.logger.error(f"Failed to schedule optimization job for {symbol}: {str(e)}")
    
    def schedule_data_update_jobs(self, symbols: List[str], interval_hours: int = 1):
        """Schedule data update jobs"""
        
        for symbol in symbols:
            try:
                job_id = self.db_manager.schedule_job(
                    symbol=symbol,
                    job_type='data_update',
                    interval_hours=interval_hours,
                    parameters={
                        'initial_period': '1mo'
                    }
                )
                self.logger.info(f"Scheduled data update job for {symbol}: {job_id}")
            except Exception as e:
                self.logger.error(f"Failed to schedule data update job for {symbol}: {str(e)}")
    
    def schedule_cleanup_job(self, interval_hours: int = 720):  # Monthly
        """Schedule database cleanup job"""
        
        try:
            job_id = self.db_manager.schedule_job(
                symbol='SYSTEM',
                job_type='cleanup',
                interval_hours=interval_hours,
                parameters={
                    'days_to_keep': 90
                }
            )
            self.logger.info(f"Scheduled cleanup job: {job_id}")
        except Exception as e:
            self.logger.error(f"Failed to schedule cleanup job: {str(e)}")
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all scheduled jobs"""
        
        try:
            # Get all active jobs
            all_jobs = list(self.db_manager.collections['scheduled_jobs'].find({
                'active': True
            }).sort('next_run', 1))
            
            # Convert ObjectIds to strings
            for job in all_jobs:
                job['_id'] = str(job['_id'])
            
            # Count by type
            job_counts = {}
            for job in all_jobs:
                job_type = job['job_type']
                job_counts[job_type] = job_counts.get(job_type, 0) + 1
            
            # Get next jobs
            next_jobs = all_jobs[:5]  # Next 5 jobs
            
            return {
                'total_active_jobs': len(all_jobs),
                'job_counts_by_type': job_counts,
                'next_jobs': next_jobs,
                'scheduler_running': self.running
            }
            
        except Exception as e:
            self.logger.error(f"Error getting job status: {str(e)}")
            return {'error': str(e)}


def run_scheduler():
    """Standalone function to run the scheduler"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Initialize scheduler
        scheduler = JobScheduler(db_manager)
        
        logger.info("Trend Filter Pro Scheduler starting...")
        
        # Start scheduler (this will run indefinitely)
        scheduler.start(check_interval=60)  # Check every minute
        
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")
    finally:
        if 'scheduler' in locals():
            scheduler.stop()
        if 'db_manager' in locals():
            db_manager.close_connection()


if __name__ == "__main__":
    run_scheduler()