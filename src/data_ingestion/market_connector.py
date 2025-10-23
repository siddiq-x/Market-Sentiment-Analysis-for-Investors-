"""
Market data connector for stock prices and financial indicators
"""
import yfinance as yf
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import time

from src.data_ingestion.base_connector import BaseConnector, DataPoint
from config.config import config


class MarketConnector(BaseConnector):
    """Connector for market data using Yahoo Finance and Alpha Vantage"""

    def __init__(self):
        super().__init__("MarketData", {"alpha_vantage_key": config.api.alpha_vantage_key})
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        # Alpha Vantage free tier: 5 calls per minute
        self.rate_limit_delay = 12
        self.last_request_time = 0

    def connect(self) -> bool:
        """Test connection to market data sources"""
        try:
            # Test Yahoo Finance
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")

            if not test_data.empty:
                self._is_connected = True
                self.logger.info("Successfully connected to market data sources")
                return True
            else:
                self.logger.error("Market data connection test failed")
                return False

        except Exception as e:
            self.logger.error(f"Market data connection error: {str(e)}")
            return False

    def _rate_limit(self):
        """Implement rate limiting for Alpha Vantage"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def fetch_stock_data(self,
                        tickers: Optional[List[str]] = None,
                        period: str = "1d",
                        interval: str = "1h") -> List[DataPoint]:
        """Fetch stock price data using Yahoo Finance"""
        if not self._is_connected:
            if not self.connect():
                return []

        tickers = tickers or config.monitored_tickers
        data_points = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period=period, interval=interval)

                if not hist_data.empty:
                    for timestamp, row in hist_data.iterrows():
                        # Create content string with OHLCV data
                        content = (
                            f"Stock {ticker}: Open=${row['Open']:.2f}, "
                            f"High=${row['High']:.2f}, Low=${row['Low']:.2f}, "
                            f"Close=${row['Close']:.2f}, Volume={int(row['Volume'])}"
                        )

                        # Calculate price change
                        if len(hist_data) > 1:
                            prev_close = (
                                hist_data['Close'].iloc[-2]
                                if timestamp == hist_data.index[-1]
                                else None
                            )
                            price_change = (
                                (row['Close'] - prev_close) / prev_close * 100
                                if prev_close
                                else 0
                            )
                        else:
                            price_change = 0

                        data_point = DataPoint(
                            source="YahooFinance",
                            timestamp=timestamp.to_pydatetime(),
                            content=content,
                            ticker=ticker,
                            credibility_score=0.95,  # High credibility for market data
                            metadata={
                                "open": float(row['Open']),
                                "high": float(row['High']),
                                "low": float(row['Low']),
                                "close": float(row['Close']),
                                "volume": int(row['Volume']),
                                "price_change_pct": price_change,
                                "data_type": "stock_price",
                                "interval": interval
                            }
                        )
                        data_points.append(data_point)

                # Add small delay between tickers
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}")

        self.logger.info(f"Fetched {len(data_points)} market data points")
        return data_points

    def fetch_economic_indicators(self) -> List[DataPoint]:
        """Fetch economic indicators from Alpha Vantage"""
        if not self.config["alpha_vantage_key"]:
            self.logger.warning("Alpha Vantage API key not configured")
            return []

        indicators = [
            ("FEDERAL_FUNDS_RATE", "Federal Funds Rate"),
            ("CPI", "Consumer Price Index"),
            ("UNEMPLOYMENT", "Unemployment Rate"),
            ("GDP", "Gross Domestic Product")
        ]

        data_points = []

        for indicator_code, indicator_name in indicators:
            self._rate_limit()

            try:
                params = {
                    "function": "ECONOMIC_INDICATOR",
                    "indicator": indicator_code,
                    "apikey": self.config["alpha_vantage_key"],
                    "datatype": "json"
                }

                response = requests.get(self.alpha_vantage_url, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json()

                    if "data" in data:
                        for point in data["data"][:10]:  # Last 10 data points
                            content = f"{indicator_name}: {point.get('value', 'N/A')}"
                            data_point = DataPoint(
                                source="AlphaVantage",
                                timestamp=datetime.strptime(point["date"], "%Y-%m-%d"),
                                content=content,
                                ticker=None,
                                credibility_score=0.98,  # Very high for economic data
                                metadata={
                                    "indicator": indicator_name,
                                    "indicator_code": indicator_code,
                                    "value": point.get("value"),
                                    "data_type": "economic_indicator"
                                }
                            )
                            data_points.append(data_point)

            except Exception as e:
                self.logger.error(f"Error fetching {indicator_name}: {str(e)}")

        self.logger.info(f"Fetched {len(data_points)} economic indicators")
        return data_points

    def fetch_company_fundamentals(self, ticker: str) -> List[DataPoint]:
        """Fetch company fundamental data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return []

            # Create content with key fundamental metrics
            content = (f"{ticker} Fundamentals: "
                      f"Market Cap=${info.get('marketCap', 0):,}, "
                      f"P/E Ratio={info.get('trailingPE', 'N/A')}, "
                      f"Revenue=${info.get('totalRevenue', 0):,}")

            data_point = DataPoint(
                source="YahooFinance",
                timestamp=datetime.now(),
                content=content,
                ticker=ticker,
                credibility_score=0.90,
                metadata={
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "revenue": info.get("totalRevenue"),
                    "profit_margin": info.get("profitMargins"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "roe": info.get("returnOnEquity"),
                    "data_type": "fundamentals",
                    "sector": info.get("sector"),
                    "industry": info.get("industry")
                }
            )

            return [data_point]

        except Exception as e:
            self.logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
            return []

    def fetch_data(self, **kwargs) -> List[DataPoint]:
        """Main fetch method"""
        data_points = []

        # Fetch stock data
        stock_data = self.fetch_stock_data(**kwargs)
        data_points.extend(stock_data)

        # Fetch economic indicators (less frequently)
        if kwargs.get("include_economic", True):
            economic_data = self.fetch_economic_indicators()
            data_points.extend(economic_data)

        return data_points

    def is_healthy(self) -> bool:
        """Check if market data connection is healthy"""
        return self._is_connected and self.connect()
