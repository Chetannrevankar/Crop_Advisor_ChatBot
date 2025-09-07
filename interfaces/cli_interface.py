"""
Command Line Interface for Crop Advisor ChatBot.
Provides text-based interaction for users.
"""

import argparse
import traceback
import logging
from pathlib import Path
from utils.nlp_processor import NLPProcessor
from utils.response_generator import ResponseGenerator
from utils.history_manager import HistoryManager
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIInterface:
    """Command Line Interface for interacting with the chatbot."""
    
    def __init__(self, crops_db_path, regional_data_path, soil_data_path):
        """
        Initialize CLI interface with data paths.
        
        Parameters:
            crops_db_path (str): Path to crops database CSV
            regional_data_path (str): Path to regional data CSV
            soil_data_path (str): Path to soil data CSV
        """
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
                
        except Exception as e:
            logger.error(f"Failed to initialize CLI interface: {e}")
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
    
    def start_interactive_session(self, user_id="cli_user"):
        """
        Start an interactive CLI session.
        
        Parameters:
            user_id (str): User identifier for history tracking
        """
        print("=" * 60)
        print("Welcome to Crop Advisor ChatBot!")
        print("=" * 60)
        print("\nI can help you with:")
        print("- Crop disease diagnosis and prevention")
        print("- Cultivation advice for various crops")
        print("- Regional crop compatibility information")
        print("\nAvailable commands:")
        print("'history' - View your query history")
        print("'stats' - View your history statistics")
        print("'clear' - Clear your history")
        print("'reset' - COMPLETELY reset ALL history (use with caution!)")
        print("'debug on/off' - Toggle debug mode")
        print("'quit' or 'exit' - End the session")
        print("=" * 60)
        
        debug_mode = False
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Thank you for using Crop Advisor ChatBot. Goodbye!")
                    break
                elif user_input.lower() == 'history':
                    self._show_history(user_id)
                    continue
                elif user_input.lower() == 'stats':
                    self._show_stats(user_id)
                    continue
                elif user_input.lower() == 'clear':
                    self._clear_history(user_id)
                    continue
                elif user_input.lower() == 'reset':
                    self._reset_all_history()
                    continue
                elif user_input.lower() == 'debug on':
                    debug_mode = True
                    print("Debug mode enabled")
                    continue
                elif user_input.lower() == 'debug off':
                    debug_mode = False
                    print("Debug mode disabled")
                    continue
                elif not user_input:
                    continue
                
                # Process query
                response = self.process_query(user_input, user_id, debug_mode)
                print(f"\nBot: {response}")
                
            except KeyboardInterrupt:
                print("\n\nSession ended. Thank you for using Crop Advisor ChatBot!")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                print("Please try again with a different query.")
                logger.error(f"Session error: {e}", exc_info=True)
    
    def process_query(self, query_text, user_id, debug_mode=False):
        """
        Process a single query and return response.
        
        Parameters:
            query_text (str): User query text
            user_id (str): User identifier for history tracking
            debug_mode (bool): Whether to show debug information
            
        Returns:
            str: Generated response
        """
        try:
            # Extract entities from query
            entities = self.nlp_processor.extract_entities(query_text)
            
            # Show debug information if enabled
            if debug_mode:
                print(f"DEBUG: Extracted entities - {entities}")
            
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
                    if debug_mode:
                        print(f"DEBUG: Similar symptom indices - {similar_symptom_indices}")
                except Exception as e:
                    logger.warning(f"Error finding similar symptoms: {e}")
                    if debug_mode:
                        print(f"DEBUG: Error in similarity search: {e}")
            
            # Generate response
            response = self.response_generator.generate_response(
                entities, similar_symptom_indices
            )
            
            # Save to history
            self.history_manager.save_query(
                user_id, query_text, response, entities
            )
            
            return response
            
        except Exception as e:
            # Get detailed error information
            error_msg = f"❌ An error occurred while processing your query: {str(e)}"
            error_msg += "\nPlease try again with a different query or contact support if the issue persists."
            
            if debug_mode:
                error_traceback = traceback.format_exc()
                error_msg += f"\n\nDEBUG: Full error traceback:\n{error_traceback}"
            
            logger.error(f"Query processing error: {e}", exc_info=True)
            return error_msg
    
    def _show_history(self, user_id):
        """Display user query history."""
        try:
            history = self.history_manager.get_user_history(user_id)
            
            if not history:
                print("No history found.")
                return
            
            print("\nYour query history:")
            print("-" * 50)
            
            for i, item in enumerate(history, 1):
                print(f"{i}. [{item['timestamp']}]")
                print(f"   Query: {item['query']}")
                # Show first 150 characters of response
                response_preview = item['response'][:150] + "..." if len(item['response']) > 150 else item['response']
                print(f"   Response: {response_preview}")
                print()
                
        except Exception as e:
            print(f"Error retrieving history: {e}")
            logger.error(f"History retrieval error: {e}")
    
    def _show_stats(self, user_id):
        """Display user history statistics."""
        try:
            stats = self.history_manager.get_history_stats(user_id)
            
            if not stats or stats.get('total_queries', 0) == 0:
                print("No history statistics available.")
                return
            
            print("\nYour history statistics:")
            print("-" * 30)
            print(f"Total queries: {stats.get('total_queries', 0)}")
            print(f"First query: {stats.get('first_query', 'N/A')}")
            print(f"Last query: {stats.get('last_query', 'N/A')}")
            
        except Exception as e:
            print(f"Error retrieving statistics: {e}")
            logger.error(f"Statistics retrieval error: {e}")
    
    def _clear_history(self, user_id):
        """Clear user's history with confirmation."""
        try:
            confirmation = input("Are you sure you want to clear your history? (yes/no): ").strip().lower()
            
            if confirmation in ['yes', 'y']:
                success = self.history_manager.clear_user_history(user_id)
                if success:
                    print("Your history has been cleared.")
                else:
                    print("No history found to clear.")
            else:
                print("History clearance cancelled.")
                
        except Exception as e:
            print(f"Error clearing history: {e}")
            logger.error(f"History clearance error: {e}")
    
    def _reset_all_history(self):
        """Reset ALL user history with confirmation."""
        try:
            print("WARNING: This will delete ALL history for ALL users!")
            confirmation = input("Are you absolutely sure? (type 'DELETE ALL' to confirm): ").strip()
            
            if confirmation == 'DELETE ALL':
                success = self.history_manager.reset_all_history()
                if success:
                    print("All history has been reset.")
                else:
                    print("Error resetting history.")
            else:
                print("History reset cancelled.")
                
        except Exception as e:
            print(f"Error resetting history: {e}")
            logger.error(f"History reset error: {e}")

def main():
    """Main function to run CLI interface."""
    parser = argparse.ArgumentParser(description='Crop Advisor ChatBot CLI')
    parser.add_argument('--user', default='cli_user', help='User ID for history tracking')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    try:
        # Initialize and start CLI interface
        cli = CLIInterface(
            crops_db_path="data/crops_database.csv",
            regional_data_path="data/regional_data.csv",
            soil_data_path="data/soil_data.csv"
        )
        
        print(f"Starting session for user: {args.user}")
        if args.debug:
            print("Debug mode enabled")
        
        cli.start_interactive_session(user_id=args.user)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all data files exist in the data/ directory.")
        logger.error(f"File not found: {e}")
    except Exception as e:
        print(f"Failed to start Crop Advisor ChatBot: {e}")
        logger.error(f"Failed to start application: {e}", exc_info=True)

if __name__ == "__main__":
    main()