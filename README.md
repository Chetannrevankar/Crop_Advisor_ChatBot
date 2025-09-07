<div align="center">

# 🌱 Crop Advisor Bot

</div>

Crop Advisor Bot is an **AI powered assistant** designed to help farmers diagnose crop diseases, get soil and region based advice, and receive step by step cultivation guidance. By combining **machine learning, NLP techniques, and domain specific agricultural data**, it helps optimize farming practices, prevent crop losses, and improve yield.

## Features ✨

* 🤖 **AI Powered Diagnosis**: Instant crop disease identification and treatment recommendations.
* 🌾 **Multi-Crop Support**: 100+ Indian crops with detailed cultivation guidance.
* 📍 **Regional Advice**: Location specific farming recommendations.
* 🌱 **Soil Compatibility**: Soil type based crop suggestions.
* 💬 **Dual Interface**: CLI and Telegram bot support.
* 📊 **History Management**: Query history and statistics tracking.

## Data Sources 📊

* [Crops Database](https://github.com/Chetannrevankar/Crop_Advisor_ChatBot/blob/main/data/crops_database.csv) – 145+ crop disease combinations with symptoms and treatments
* [Regional Data](https://github.com/Chetannrevankar/Crop_Advisor_ChatBot/blob/main/data/regional_data.csv) – 138 Indian regions with crop suitability ratings
* [Soil Data](https://github.com/Chetannrevankar/Crop_Advisor_ChatBot/blob/main/data/soil_data.csv) – 141 soil types with crop compatibility advice
* Agricultural knowledge base validated from trusted sources

## Technology Stack 🛠️

* **Python 3.8+**
* **pandas** – Data processing and management
* **Transformers (DistilBERT)** – NLP embeddings
* **FAISS** – Similarity search for symptoms
* **SQLite** – User history storage
* **python-telegram-bot** – Telegram interface

### Machine Learning & NLP

* **ML Algorithms**: Symptom similarity matching, FAISS-based nearest neighbor search
* **LLM Model**: DistilBERT for embeddings & entity extraction
* **NLP Techniques**: Intent classification, entity recognition, synonym & variation handling

## Installation ⚙️

```bash
# Clone the repository
git clone https://github.com/Chetannrevankar/Crop_Advisor_ChatBot.git
cd Crop_Advisor_ChatBot

# Install dependencies
pip install -r requirements.txt
```

## Quick Start 🚀

### CLI Mode

```bash
python main.py --mode cli
```

### Telegram Bot Mode

1. Get bot token from **@BotFather**
2. Add token to `config.py`
3. Run:

```bash
python main.py --mode telegram
```

## Outputs 📂

* [CLI Interface Outputs](https://github.com/Chetannrevankar/Crop_Advisor_ChatBot/tree/main/outputs/cli)
* [Telegram Interface Outputs](https://github.com/Chetannrevankar/Crop_Advisor_ChatBot/tree/main/outputs/telegram_interface)

## License 📄

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## Contact 💬

For support and questions:

* 📧 [Email](mailto:chetannrevankar00001@gmail.com)
* 🔗 [LinkedIn](https://www.linkedin.com/in/chetannrevankar)

## Acknowledgments 🙏

* Indian Council of Agricultural Research (ICAR) for research and guidance
* Krishi Vigyan Kendras for agricultural data
* Hugging Face for transformer models
* Telegram Bot API for messaging platform

🌱 Helping farmers grow better, one query at a time.
