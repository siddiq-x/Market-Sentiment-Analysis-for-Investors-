# Advanced Market Sentiment Analysis System

An end-to-end solution that ingests unstructured (news articles, social media) and structured (stock prices, indicators) data, performs real-time sentiment analysis, integrates multimodal data, predicts stock price movements, and presents results via an explainable dashboard.

## 🏗️ System Architecture

### Components
- **Data Ingestion Layer**: Financial News APIs, Twitter API, SEC Filings, Stock Market Data
- **Data Preprocessing**: Bot detection, credibility scoring, NER, text cleaning
- **Sentiment Analysis**: FinBERT-based sentiment extraction
- **Multimodal Fusion**: LSTM with attention mechanism for price prediction
- **Real-time Processing**: Streaming pipeline simulation
- **Explainable Dashboard**: SHAP-powered insights and visualizations
- **Continuous Learning**: Model retraining pipeline

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the system:
```bash
python main.py
```

4. Access the dashboard:
```
http://localhost:8080
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

## 🔧 Configuration

The system uses environment variables for API keys and configuration. See `.env.example` for required variables.

## 📊 Features

- Real-time sentiment analysis of financial news and social media
- Multi-source data integration (news, social media, market data)
- Explainable AI predictions with SHAP values
- Interactive dashboard with historical analysis
- Continuous model learning and drift detection
- Bot detection and source credibility scoring

## 🧪 Testing

```bash
pytest tests/
```

## 📈 Performance

The system is designed to handle:
- 1000+ news articles per hour
- Real-time social media streams
- Sub-second prediction latency
- 95%+ uptime with fault tolerance

