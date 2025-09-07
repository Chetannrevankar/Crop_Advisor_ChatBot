"""
Telegram Bot Interface for Crop Advisor ChatBot.
Provides interactive messaging through Telegram.
"""

import logging
import os
from pathlib import Path
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    ContextTypes, ConversationHandler, filters
)
import pandas as pd

from utils.nlp_processor import NLPProcessor
from utils.response_generator import ResponseGenerator
from utils.history_manager import HistoryManager

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
CROP, SYMPTOM, REGION, SOIL = range(4)

class TelegramBot:
    """Telegram Bot interface for Crop Advisor ChatBot."""
    
    def __init__(self, token, crops_db_path, regional_data_path, soil_data_path):
        """
        Initialize Telegram bot with data paths.
        
        Parameters:
            token (str): Telegram bot token
            crops_db_path (str): Path to crops database CSV
            regional_data_path (str): Path to regional data CSV
            soil_data_path (str): Path to soil data CSV
        """
        self.token = token
        
        try:
            # Load data with error handling
            self.crops_db = self._load_data_file(crops_db_path, "crops database")
            self.regional_data = self._load_data_file(regional_data_path, "regional data")
            self.soil_data = self._load_data_file(soil_data_path, "soil data")
            
            # Initialize components
            self.nlp_processor = NLPProcessor()
            self.response_generator = ResponseGenerator(
                self.crops_db, self.regional_data, self.soil_data
            )
            self.history_manager = HistoryManager()
            
            # Build FAISS index for symptoms
            if 'symptom' in self.crops_db.columns:
                symptom_descriptions = self.crops_db['symptom'].dropna().tolist()
                if symptom_descriptions:
                    self.nlp_processor.build_faiss_index(symptom_descriptions)
                    logger.info(f"Built FAISS index with {len(symptom_descriptions)} symptoms")
                else:
                    logger.warning("No symptom descriptions found for FAISS index")
            else:
                logger.warning("'symptom' column not found in crops database")
            
            # User sessions
            self.user_sessions = {}
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            raise
    
    def _load_data_file(self, file_path, description):
        """Load data file with proper error handling."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"{description} file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {description} with {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {description} from {file_path}: {e}")
            raise
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send welcome message when command /start is issued."""
        user_id = str(update.effective_user.id)
        
        welcome_message = (
            "🌱 Welcome to Crop Advisor ChatBot! 🌱\n\n"
            "I can help you with:\n"
            "• Crop disease diagnosis and prevention\n"
            "• Cultivation advice for various crops\n"
            "• Regional crop compatibility information\n\n"
            "Available commands:\n"
            "/help - Show help message\n"
            "/history - View your query history\n"
            "/stats - View your history statistics\n"
            "/clear - Clear your history\n"
            "/reset - COMPLETELY reset ALL history (admin only)\n\n"
            "You can also just type your question naturally, like:\n"
            "• My tomato plants have yellow leaves\n"
            "• How to grow rice in clay soil\n"
            "• What crops grow well in Karnataka\n\n"
            "Disclaimer: This is AI generated advice. Please consult agricultural officers for confirmed diagnosis."
        )
        
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send help message when command /help is issued."""
        help_message = (
            "🌱 Crop Advisor ChatBot - Help\n\n"
            "Available commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/history - Show your query history\n"
            "/stats - Show your history statistics\n"
            "/clear - Clear your history\n"
            "/reset - COMPLETELY reset ALL history (admin only)\n\n"
            "You can ask questions like:\n"
            "• Disease diagnosis: 'My tomato plants have yellow leaves in Karnataka'\n"
            "• Cultivation advice: 'How to grow cotton in black soil'\n"
            "• Regional compatibility: 'What crops grow well in Maharashtra'\n"
            "• Soil advice: 'Best crops for sandy soil in coastal areas'\n\n"
            "Tip: Be specific about crop name, symptoms, and region for better results!"
        )
        
        await update.message.reply_text(help_message)
    
    async def show_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user query history."""
        user_id = str(update.effective_user.id)
        try:
            history = self.history_manager.get_user_history(user_id)
            
            if not history:
                await update.message.reply_text("📝 You don't have any query history yet.")
                return
            
            response = "📋 Your Recent Queries:\n\n"
            for i, item in enumerate(history[:5], 1):  # Show last 5 queries
                response += f"{i}. [{item['timestamp']}]\n"
                response += f"   Q: {item['query']}\n"
                response += f"   A: {item['response'][:80]}...\n\n"
            
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error showing history: {e}")
            await update.message.reply_text("❌ Error retrieving your history. Please try again.")
    
    async def show_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user history statistics."""
        user_id = str(update.effective_user.id)
        try:
            stats = self.history_manager.get_history_stats(user_id)
            
            if not stats or stats.get('total_queries', 0) == 0:
                await update.message.reply_text("📊 You don't have any query history yet.")
                return
            
            response = (
                f"📊 Your History Statistics:\n\n"
                f"• Total queries: {stats.get('total_queries', 0)}\n"
                f"• First query: {stats.get('first_query', 'N/A')}\n"
                f"• Last query: {stats.get('last_query', 'N/A')}"
            )
            
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error showing stats: {e}")
            await update.message.reply_text("❌ Error retrieving statistics. Please try again.")
    
    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clear user query history with confirmation."""
        user_id = str(update.effective_user.id)
        
        # Create confirmation keyboard
        reply_keyboard = [['✅ Yes, clear my history', '❌ No, keep my history']]
        
        await update.message.reply_text(
            "🗑️ Are you sure you want to clear ALL your query history?",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, 
                one_time_keyboard=True,
                input_field_placeholder='Clear history?'
            )
        )
        
        # Store user ID for confirmation handling
        context.user_data['awaiting_clear_confirmation'] = True
        return CROP
    
    async def handle_clear_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle confirmation for clearing history."""
        user_id = str(update.effective_user.id)
        user_response = update.message.text
        
        try:
            if user_response == '✅ Yes, clear my history':
                success = self.history_manager.clear_user_history(user_id)
                if success:
                    await update.message.reply_text(
                        "✅ Your history has been cleared successfully.",
                        reply_markup=ReplyKeyboardRemove()
                    )
                else:
                    await update.message.reply_text(
                        "ℹ️ No history found to clear.",
                        reply_markup=ReplyKeyboardRemove()
                    )
            else:
                await update.message.reply_text(
                    "❌ History clearance cancelled.",
                    reply_markup=ReplyKeyboardRemove()
                )
            
            # Clear the confirmation flag
            context.user_data['awaiting_clear_confirmation'] = False
            return ConversationHandler.END
            
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            await update.message.reply_text(
                "❌ Error clearing history. Please try again.",
                reply_markup=ReplyKeyboardRemove()
            )
            return ConversationHandler.END
    
    async def reset_all_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reset ALL history (admin function)."""
        user_id = str(update.effective_user.id)
        
        # Admin check using environment variable
        admin_user_ids = os.getenv('ADMIN_USER_IDS', '').split(',')
        if user_id not in admin_user_ids:
            await update.message.reply_text("⛔ This command is for administrators only.")
            return
        
        # Create confirmation keyboard
        reply_keyboard = [['🗑️ DELETE ALL HISTORY', '❌ Cancel']]
        
        await update.message.reply_text(
            "🚨 WARNING: This will delete ALL history for ALL users!\n\n"
            "Type 'DELETE ALL HISTORY' to confirm or 'Cancel' to abort.",
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, 
                one_time_keyboard=True,
                input_field_placeholder='Confirm reset?'
            )
        )
        
        # Store admin ID for confirmation handling
        context.user_data['awaiting_reset_confirmation'] = True
        return CROP
    
    async def handle_reset_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle confirmation for resetting all history."""
        user_response = update.message.text
        
        try:
            if user_response == '🗑️ DELETE ALL HISTORY':
                success = self.history_manager.reset_all_history()
                if success:
                    await update.message.reply_text(
                        "✅ All history has been reset successfully.",
                        reply_markup=ReplyKeyboardRemove()
                    )
                else:
                    await update.message.reply_text(
                        "❌ Error resetting history.",
                        reply_markup=ReplyKeyboardRemove()
                    )
            else:
                await update.message.reply_text(
                    "❌ History reset cancelled.",
                    reply_markup=ReplyKeyboardRemove()
                )
            
            # Clear the confirmation flag
            context.user_data['awaiting_reset_confirmation'] = False
            return ConversationHandler.END
            
        except Exception as e:
            logger.error(f"Error resetting history: {e}")
            await update.message.reply_text(
                "❌ Error resetting history. Please try again.",
                reply_markup=ReplyKeyboardRemove()
            )
            return ConversationHandler.END
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        # Check if we're waiting for a confirmation
        if context.user_data.get('awaiting_clear_confirmation'):
            return await self.handle_clear_confirmation(update, context)
        
        if context.user_data.get('awaiting_reset_confirmation'):
            return await self.handle_reset_confirmation(update, context)
        
        user_id = str(update.effective_user.id)
        query_text = update.message.text
        
        try:
            # Show typing action
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            # Process query
            entities = self.nlp_processor.extract_entities(query_text)
            logger.info(f"User {user_id} query: '{query_text}' -> Entities: {entities}")
            
            # Find similar symptoms if it's a diagnosis query
            similar_symptom_indices = None
            if (entities.get('intent') == 'disease_diagnosis' and 
                entities.get('symptoms') and 
                hasattr(self.nlp_processor, 'faiss_index') and 
                self.nlp_processor.faiss_index is not None):
                
                try:
                    similar_symptom_indices = self.nlp_processor.find_similar_symptoms(
                        ' '.join(entities['symptoms'])
                    )
                    logger.debug(f"Similar symptom indices: {similar_symptom_indices}")
                except Exception as e:
                    logger.warning(f"Error finding similar symptoms: {e}")
            
            # Generate response
            response = self.response_generator.generate_response(
                entities, similar_symptom_indices
            )
            
            # Save to history
            self.history_manager.save_query(
                user_id, query_text, response, entities
            )
            
            # Send response (split if too long for Telegram)
            if len(response) > 4096:  # Telegram message limit
                for x in range(0, len(response), 4096):
                    await update.message.reply_text(response[x:x+4096])
            else:
                await update.message.reply_text(response)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            error_message = (
                "❌ Sorry, I encountered an error processing your request.\n\n"
                "Please try:\n"
                "• Rephrasing your question\n"
                "• Being more specific about crop and symptoms\n"
                "• Trying again in a moment\n\n"
                "If the problem persists, please contact support."
            )
            await update.message.reply_text(error_message)
    
    def setup_handlers(self, application):
        """Set up all Telegram bot handlers."""
        try:
            # Add command handlers
            application.add_handler(CommandHandler("start", self.start))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("history", self.show_history))
            application.add_handler(CommandHandler("stats", self.show_stats))
            application.add_handler(CommandHandler("clear", self.clear_history))
            application.add_handler(CommandHandler("reset", self.reset_all_history))
            
            # Add message handler
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            logger.info("Telegram bot handlers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up handlers: {e}")
            raise
    
    def run(self):
        """Run the Telegram bot."""
        try:
            # Create Application
            application = Application.builder().token(self.token).build()
            
            # Set up handlers
            self.setup_handlers(application)
            
            # Start the bot
            logger.info("Starting Telegram bot...")
            print("🌱 Crop Advisor Telegram Bot is running...")
            print("Press Ctrl+C to stop the bot")
            
            application.run_polling()
            
        except Exception as e:
            logger.error(f"Failed to run Telegram bot: {e}")
            raise

def main():
    """Main function to run Telegram bot."""
    try:
        from config import TELEGRAM_BOT_TOKEN
        
        if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
            raise ValueError("Please set your Telegram bot token in config.py")
        
        # Initialize and run Telegram bot
        bot = TelegramBot(
            token=TELEGRAM_BOT_TOKEN,
            crops_db_path="data/crops_database.csv",
            regional_data_path="data/regional_data.csv",
            soil_data_path="data/soil_data.csv"
        )
        
        bot.run()
        
    except ImportError:
        print("Error: config.py not found. Please create config.py with your Telegram bot token.")
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}\nPlease ensure all data files exist in the data/ directory.")
    except Exception as e:
        print(f"Failed to start Telegram bot: {e}")
        logger.error(f"Failed to start Telegram bot: {e}", exc_info=True)

if __name__ == "__main__":
    main()