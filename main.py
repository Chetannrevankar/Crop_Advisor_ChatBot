"""
Main entry point for Crop Advisor ChatBot.
Provides both CLI and Telegram interfaces.
"""

import argparse
import sys
from interfaces.cli_interface import CLIInterface
from interfaces.telegram_bot import TelegramBot
from config import TELEGRAM_BOT_TOKEN

def main():
    """Main function to run the Crop Advisor ChatBot."""
    parser = argparse.ArgumentParser(description='Crop Advisor ChatBot')
    parser.add_argument(
        '--mode', 
        choices=['cli', 'telegram'], 
        default='cli',
        help='Run mode: cli (command line) or telegram (bot)'
    )
    parser.add_argument('--user', default='default_user', help='User ID for history tracking')
    args = parser.parse_args()
    
    if args.mode == 'cli':
        # Run CLI interface
        cli = CLIInterface(
            crops_db_path="data/crops_database.csv",
            regional_data_path="data/regional_data.csv",
            soil_data_path="data/soil_data.csv"
        )
        cli.start_interactive_session(user_id=args.user)
    
    elif args.mode == 'telegram':
        # Run Telegram bot
        if TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
            print("Error: Please set your Telegram bot token in config.py")
            sys.exit(1)
            
        bot = TelegramBot(
            token=TELEGRAM_BOT_TOKEN,
            crops_db_path="data/crops_database.csv",
            regional_data_path="data/regional_data.csv",
            soil_data_path="data/soil_data.csv"
        )
        bot.run()

if __name__ == "__main__":
    main()