# Advanced Market Sentiment Analysis System

An end-to-end solution that ingests unstructured (news articles, social media) and structured (stock prices, indicators) data, performs real-time sentiment analysis, integrates multimodal data, predicts stock price movements, and presents results via an explainable dashboard.

## 📌 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🚀 Features

- Real-time sentiment analysis of financial news and social media
- Multi-source data integration (news, social media, market data)
- Explainable AI predictions with SHAP values
- Interactive dashboard with historical analysis
- Continuous model learning and drift detection
- Bot detection and source credibility scoring

## 🏗️ System Architecture

### Components
- **Data Ingestion Layer**: Financial News APIs, Twitter API, SEC Filings, Stock Market Data
- **Data Preprocessing**: Bot detection, credibility scoring, NER, text cleaning
- **Sentiment Analysis**: FinBERT-based sentiment extraction
- **Multimodal Fusion**: LSTM with attention mechanism for price prediction
- **Real-time Processing**: Streaming pipeline simulation
- **Explainable Dashboard**: SHAP-powered insights and visualizations
- **Continuous Learning**: Model retraining pipeline

## 💻 Installation

### Prerequisites
- Python 3.8+
- pip
- MongoDB (for data storage)
- Redis (for caching)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/siddiq-x/Market-Sentiment-Analysis-for-Investors-.git
cd Market-Sentiment-Analysis-for-Investors
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## 🔧 Configuration

Edit the `.env` file with your API keys and settings:

```bash
# API Keys
NEWS_API_KEY=your_newsapi_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Database Configuration
MONGODB_URI=mongodb://localhost:27017/sentiment_analysis
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Configuration
FINBERT_MODEL_PATH=models/finbert
SENTIMENT_THRESHOLD=0.1
```

## 🚀 Usage

### Running the Dashboard
```bash
python main.py --mode dashboard
```

### Processing Data in Batch Mode
```bash
python main.py --mode batch --input data/input.json --output results/
```

### Example API Usage
```python
from src.sentiment.ensemble_analyzer import EnsembleSentimentAnalyzer

# Initialize the analyzer
analyzer = EnsembleSentimentAnalyzer()

# Analyze text sentiment
result = analyzer.analyze("Apple stock reaches all-time high!")
print(f"Sentiment: {result['label']} (Score: {result['score']:.2f})")
```

## 📁 Project Structure

```
├── src/
│   ├── data_ingestion/     # API connectors and data sources
│   ├── preprocessing/      # Data cleaning and NER
│   ├── sentiment/          # FinBERT sentiment analysis
│   ├── fusion/            # Multimodal fusion engine
│   ├── pipeline/          # Real-time processing
│   ├── dashboard/         # Web interface and visualizations
│   └── utils/             # Shared utilities
├── models/                # Trained models and checkpoints
├── data/                  # Raw and processed data
├── config/                # Configuration files
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage report:
```bash
pytest --cov=src tests/
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.

## 📈 Performance

The system is designed to handle:
- 1000+ news articles per hour
- Real-time social media streams
- Sub-second prediction latency
- 95%+ uptime with fault tolerance

