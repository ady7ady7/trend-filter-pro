"""
Trend Filter Application - Main Entry Point
A comprehensive trading instrument trend analysis tool
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from trend_analyzer import TrendAnalyzer
from data_manager import DataManager
from config import Config
from utils import setup_logging, save_results_to_csv

def main():
    """Main application entry point"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Trend Filter Application")
    
    # Initialize configuration
    config = Config()
    
    # Get user input for analysis mode
    print("\n=== TREND FILTER APPLICATION ===")
    print("1. Analyze single instrument")
    print("2. Analyze multiple instruments from list")
    
    choice = input("\nSelect mode (1 or 2): ").strip()
    
    if choice == "1":
        # Single instrument analysis
        symbol = input("Enter instrument symbol (e.g., EURUSD=X, ^GSPC, GC=F): ").strip().upper()
        if not symbol:
            print("No symbol provided. Exiting.")
            return
        
        instruments = [symbol]
        
    elif choice == "2":
        # Multiple instruments analysis
        print("\nUsing predefined instrument list:")
        instruments = config.DEFAULT_INSTRUMENTS
        for i, instrument in enumerate(instruments, 1):
            print(f"{i}. {instrument}")
        
        use_default = input("\nUse this list? (y/n): ").strip().lower()
        if use_default != 'y':
            print("Enter instruments separated by commas:")
            custom_input = input().strip()
            if custom_input:
                instruments = [s.strip().upper() for s in custom_input.split(',')]
            else:
                print("No instruments provided. Exiting.")
                return
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Initialize components
    data_manager = DataManager()
    trend_analyzer = TrendAnalyzer()
    
    # Process each instrument
    results_summary = []
    
    for symbol in instruments:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {symbol}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Fetch and prepare data
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
                continue
            
            print(f"✅ Data fetched successfully")
            print(f"   Long-term: {len(long_term_data)} data points ({config.LONG_TERM_INTERVAL})")
            print(f"   Short-term: {len(short_term_data)} data points ({config.SHORT_TERM_INTERVAL})")
            
            # Step 2: Perform comprehensive trend analysis
            print("\n🔍 Performing trend analysis...")
            
            # Long-term analysis
            long_term_analysis = trend_analyzer.analyze_trend(
                long_term_data, 
                timeframe="long_term",
                symbol=symbol
            )
            
            # Short-term analysis  
            short_term_analysis = trend_analyzer.analyze_trend(
                short_term_data,
                timeframe="short_term", 
                symbol=symbol
            )
            
            # Step 3: Generate comprehensive report
            print("\n📋 Generating analysis report...")
            
            report = trend_analyzer.generate_comprehensive_report(
                symbol,
                long_term_analysis,
                short_term_analysis,
                long_term_data,
                short_term_data
            )
            
            # Step 4: Save results
            print("\n💾 Saving results...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_analysis_{symbol}_{timestamp}.csv"
            
            success = save_results_to_csv(report, filename)
            
            if success:
                print(f"✅ Results saved to: {filename}")
                
                # Add to summary
                results_summary.append({
                    'Symbol': symbol,
                    'Trading_Score': report.get('trading_score', 0),
                    'Market_State': report.get('market_state', 'Unknown'),
                    'Trend_Direction': report.get('trend_direction', 'Unclear'),
                    'Risk_Level': report.get('risk_assessment', 'Unknown'),
                    'Filename': filename,
                    'Analysis_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Display key results
                print(f"\n📈 ANALYSIS SUMMARY FOR {symbol}:")
                print(f"   Trading Score: {report.get('trading_score', 0)}/100")
                print(f"   Market State: {report.get('market_state', 'Unknown')}")
                print(f"   Trend Direction: {report.get('trend_direction', 'Unclear')}")
                print(f"   Risk Level: {report.get('risk_assessment', 'Unknown')}")
                
            else:
                print(f"❌ Failed to save results for {symbol}")
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            continue
    
    # Save summary report if multiple instruments
    if len(instruments) > 1 and results_summary:
        print(f"\n{'='*60}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*60}")
        
        summary_df = pd.DataFrame(results_summary)
        summary_filename = f"trend_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            summary_df.to_csv(summary_filename, index=False)
            print(f"✅ Summary report saved to: {summary_filename}")
            
            # Display top opportunities
            print("\n🎯 TOP TRADING OPPORTUNITIES:")
            top_opportunities = summary_df.nlargest(5, 'Trading_Score')
            for idx, row in top_opportunities.iterrows():
                print(f"   {row['Symbol']}: {row['Trading_Score']}/100 - {row['Market_State']}")
                
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            print(f"❌ Error saving summary: {str(e)}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed {len([r for r in results_summary if r])} instruments successfully")
    print("📁 Check the generated CSV files for detailed analysis")

if __name__ == "__main__":
    main()