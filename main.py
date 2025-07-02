"""
Enhanced Trend Filter Application
Includes parameter optimization and database integration
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from trend_analyzer import TrendAnalyzer
from data_manager import DataManager
from config import Config
from utils import setup_logging, save_results_to_csv
from database_manager import DatabaseManager
from parameter_optimizer import ParameterOptimizer
from scheduler import JobScheduler

def main():
    """Enhanced main application with new features"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Enhanced Trend Filter Application")
    
    # Initialize configuration
    config = Config()
    
    # Initialize database connection
    try:
        db_manager = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        print("❌ Database connection failed. Running in offline mode.")
        db_manager = None
    
    # Get user input for enhanced features
    print("\n=== ENHANCED TREND FILTER APPLICATION ===")
    print("1. Analyze single instrument")
    print("2. Analyze multiple instruments from list")
    print("3. Optimize parameters for instrument")
    print("4. Setup scheduled jobs")
    print("5. View database insights")
    print("6. Run scheduler daemon")
    
    try:
        choice = input("\nSelect mode (1-6): ").strip()
        print(f"DEBUG: You entered: '{choice}'")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        return
    
    if choice == "1":
        run_single_analysis(db_manager, config)
    elif choice == "2":
        run_batch_analysis(db_manager, config)
    elif choice == "3":
        run_parameter_optimization(db_manager)
    elif choice == "4":
        setup_scheduled_jobs(db_manager, config)
    elif choice == "5":
        view_database_insights(db_manager)
    elif choice == "6":
        run_scheduler_daemon(db_manager)
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Cleanup
    if db_manager:
        db_manager.close_connection()

def run_single_analysis(db_manager: DatabaseManager, config: Config):
    """Run single instrument analysis with database integration"""
    
    try:
        symbol = input("Enter instrument symbol (e.g., EURUSD=X, ^GSPC, GC=F): ").strip().upper()
        print(f"DEBUG: Symbol entered: '{symbol}'")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        return
    
    if not symbol:
        print("No symbol provided. Exiting.")
        return
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {symbol}")
    print(f"{'='*60}")
    
    # Initialize components
    data_manager = DataManager()
    trend_analyzer = TrendAnalyzer()
    
    try:
        # Check for optimal parameters in database
        if db_manager:
            print("🔍 Checking for optimal parameters...")
            optimal_params = db_manager.get_optimal_parameters(symbol, 'medium_term')
            if optimal_params:
                print("✅ Found optimal parameters, applying to analysis")
                trend_analyzer.ema_periods = optimal_params['parameters']['ema_periods']
            else:
                print("ℹ️  No optimal parameters found, using defaults")
        
        # Fetch data
        print("📊 Fetching market data...")
        long_term_data = data_manager.fetch_data(
            symbol, 
            period=config.LONG_TERM_PERIOD,
            interval=config.LONG_TERM_INTERVAL
        )
        
        short_term_data = data_manager.fetch_data(
            symbol,
            period=config.SHORT_TERM_PERIOD, 
            interval=config.SHORT_TERM_INTERVAL
        )
        
        if long_term_data.empty or short_term_data.empty:
            print(f"❌ Failed to fetch data for {symbol}")
            return
        
        print(f"✅ Data fetched successfully")
        print(f"   Long-term: {len(long_term_data)} data points ({config.LONG_TERM_INTERVAL})")
        print(f"   Short-term: {len(short_term_data)} data points ({config.SHORT_TERM_INTERVAL})")
        
        # Perform analysis
        print("\n🔍 Performing trend analysis...")
        
        long_term_analysis = trend_analyzer.analyze_trend(
            long_term_data, 
            timeframe="long_term",
            symbol=symbol
        )
        
        short_term_analysis = trend_analyzer.analyze_trend(
            short_term_data,
            timeframe="short_term", 
            symbol=symbol
        )
        
        # Generate report
        print("\n📋 Generating analysis report...")
        
        report = trend_analyzer.generate_comprehensive_report(
            symbol,
            long_term_analysis,
            short_term_analysis,
            long_term_data,
            short_term_data
        )
        
        # Display results
        print(f"\n📈 ANALYSIS SUMMARY FOR {symbol}:")
        print(f"   Trading Score: {report.get('trading_score', 0)}/100")
        print(f"   Market State: {report.get('market_state', 'Unknown')}")
        print(f"   Trend Direction: {report.get('trend_direction', 'Unclear')}")
        print(f"   Risk Level: {report.get('risk_assessment', 'Unknown')}")
        
        # Save to database
        if db_manager:
            print("\n💾 Saving to database...")
            try:
                result_id = db_manager.store_analysis_result(report)
                print(f"✅ Results saved to database: {result_id}")
                
                # Also store market data
                data_points = db_manager.bulk_store_market_data(symbol, short_term_data)
                print(f"✅ Stored {data_points} market data points")
                
            except Exception as e:
                print(f"❌ Database save failed: {str(e)}")
        
        # Save CSV as backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trend_analysis_{symbol}_{timestamp}.csv"
        success = save_results_to_csv(report, filename)
        
        if success:
            print(f"✅ CSV backup saved: {filename}")
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")

def run_batch_analysis(db_manager: DatabaseManager, config: Config):
    """Run batch analysis with database integration"""
    
    print("\nUsing predefined instrument list:")
    instruments = config.DEFAULT_INSTRUMENTS
    for i, instrument in enumerate(instruments, 1):
        print(f"{i}. {instrument}")
    
    use_default = input("\nUse this list? (y/n): ").strip().lower()
    if use_default != 'y':
        print("Enter instruments separated by commas:")
        try:
            custom_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return
        if custom_input:
            instruments = [s.strip().upper() for s in custom_input.split(',')]
        else:
            print("No instruments provided. Exiting.")
            return
    
    # Initialize components
    data_manager = DataManager()
    trend_analyzer = TrendAnalyzer()
    
    results_summary = []
    
    for symbol in instruments:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {symbol}")
        print(f"{'='*60}")
        
        try:
            # Check for optimal parameters
            if db_manager:
                optimal_params = db_manager.get_optimal_parameters(symbol, 'medium_term')
                if optimal_params:
                    trend_analyzer.ema_periods = optimal_params['parameters']['ema_periods']
            
            # Continue with normal analysis...
            # (Same as single analysis but in loop)
            # [Rest of analysis code here]
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*60}")

def run_parameter_optimization(db_manager: DatabaseManager):
    """Run parameter optimization for an instrument"""
    
    if not db_manager:
        print("❌ Database connection required for parameter optimization")
        return
    
    try:
        symbol = input("Enter symbol for optimization: ").strip().upper()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        return
    
    if not symbol:
        print("No symbol provided. Exiting.")
        return
    
    print(f"\n🔧 PARAMETER OPTIMIZATION FOR {symbol}")
    print("=" * 60)
    
    optimizer = ParameterOptimizer()
    
    try:
        print("🚀 Starting optimization (this may take several minutes)...")
        
        # Run optimization for multiple timeframes
        optimization_results = optimizer.optimize_multiple_timeframes(symbol)
        
        print(f"\n📊 OPTIMIZATION RESULTS:")
        print("=" * 50)
        
        for timeframe, result in optimization_results.items():
            print(f"\n{timeframe.upper()} TIMEFRAME:")
            print(f"  Validation Score: {result.validation_score:.2f}")
            print(f"  Confidence Level: {result.confidence_level}")
            print(f"  Overfitting Score: {result.overfitting_score:.2f}")
            print(f"  Sample Size: {result.sample_size}")
            
            # Show best parameters
            params = result.best_parameters
            print(f"  Best EMAs: {params.ema_periods}")
            print(f"  RSI Period: {params.rsi_period}")
            print(f"  Slope Threshold: {params.slope_threshold}")
            
            # Store in database
            result_id = db_manager.store_optimal_parameters(result, symbol, timeframe)
            print(f"  ✅ Stored in database: {result_id}")
        
        # Test robustness
        print(f"\n🧪 Testing parameter robustness...")
        best_result = max(optimization_results.values(), key=lambda x: x.validation_score)
        
        robustness = optimizer.validate_parameters_robustness(
            best_result.best_parameters, 
            symbol, 
            ['1mo', '2mo', '3mo', '6mo']
        )
        
        print(f"  Mean Score: {robustness.get('mean_score', 0):.2f}")
        print(f"  Consistency: {robustness.get('consistency', 0):.2f}")
        print(f"  Test Periods: {robustness.get('test_periods', 0)}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")

def setup_scheduled_jobs(db_manager: DatabaseManager, config: Config):
    """Setup scheduled jobs for automated analysis"""
    
    if not db_manager:
        print("❌ Database connection required for scheduled jobs")
        return
    
    print("\n⏰ SETUP SCHEDULED JOBS")
    print("=" * 40)
    
    scheduler = JobScheduler(db_manager)
    
    # Get symbols to schedule
    print("Choose symbols for scheduling:")
    print("1. Use default instrument list")
    print("2. Enter custom symbols")
    
    choice = input("Choice (1 or 2): ").strip()
    
    if choice == "1":
        symbols = config.DEFAULT_INSTRUMENTS[:5]  # Limit to avoid overload
    elif choice == "2":
        custom_input = input("Enter symbols (comma-separated): ").strip()
        symbols = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
    else:
        print("Invalid choice")
        return
    
    if not symbols:
        print("No symbols provided")
        return
    
    print(f"\nScheduling jobs for: {', '.join(symbols)}")
    
    # Schedule different job types
    try:
        # Analysis jobs (every 4 hours)
        print("\n📊 Scheduling analysis jobs (every 4 hours)...")
        scheduler.schedule_analysis_jobs(symbols, interval_hours=4)
        
        # Data update jobs (every hour)
        print("📈 Scheduling data update jobs (every hour)...")
        scheduler.schedule_data_update_jobs(symbols, interval_hours=1)
        
        # Optimization jobs (weekly)
        print("🔧 Scheduling optimization jobs (weekly)...")
        scheduler.schedule_optimization_jobs(symbols, interval_hours=168)
        
        # Cleanup job (monthly)
        print("🧹 Scheduling cleanup job (monthly)...")
        scheduler.schedule_cleanup_job(interval_hours=720)
        
        print("\n✅ All jobs scheduled successfully!")
        print("\nTo start the scheduler daemon, run: python main_enhanced.py (option 6)")
        
    except Exception as e:
        print(f"❌ Error scheduling jobs: {str(e)}")

def view_database_insights(db_manager: DatabaseManager):
    """View insights from the database"""
    
    if not db_manager:
        print("❌ Database connection required")
        return
    
    print("\n📊 DATABASE INSIGHTS")
    print("=" * 40)
    
    try:
        # Get best parameters by period
        print("\n🏆 BEST PARAMETERS (Last 7 Days):")
        best_params_week = db_manager.get_best_parameters_by_period(7)
        
        for result in best_params_week[:5]:  # Top 5
            symbol = result['symbol']
            params = result['best_parameters']
            print(f"\n{symbol}:")
            print(f"  Score: {params['validation_score']:.2f}")