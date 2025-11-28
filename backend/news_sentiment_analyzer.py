"""
News & Sentiment Analyzer for Indian Stock Market
Features:
- Real-time news fetching from multiple sources
- Sentiment analysis (positive/negative/neutral)
- Index direction detection (NIFTY/BANKNIFTY)
- Sector-specific news impact
- Stock-specific news alerts
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """News article data"""
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    symbols: List[str]  # Related symbols
    sentiment_score: float  # -1 to 1
    relevance: float  # 0 to 1
    category: str  # market/stock/sector/policy


class NewsSentimentAnalyzer:
    """
    Fetch and analyze news from Indian market sources
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize news analyzer
        
        Args:
            api_keys: Dict with keys: 'newsapi', 'alphavantage', etc.
        """
        self.api_keys = api_keys or {}
        
        # Sentiment keywords (simplified - can be enhanced with ML models)
        self.positive_keywords = {
            'surge', 'rally', 'gains', 'profit', 'growth', 'bullish', 'upgrade',
            'buy', 'strong', 'record', 'high', 'beat', 'positive', 'recovery',
            'expansion', 'breakthrough', 'achievement', 'success', 'outperform'
        }
        
        self.negative_keywords = {
            'fall', 'drop', 'crash', 'loss', 'decline', 'bearish', 'downgrade',
            'sell', 'weak', 'low', 'miss', 'negative', 'recession', 'concern',
            'risk', 'warning', 'threat', 'failure', 'underperform', 'slump'
        }
        
        # Indian stock symbols mapping
        self.symbol_mappings = {
            'reliance': 'RELIANCE-EQ',
            'tcs': 'TCS-EQ',
            'infosys': 'INFY-EQ',
            'hdfc bank': 'HDFCBANK-EQ',
            'icici bank': 'ICICIBANK-EQ',
            'state bank': 'SBIN-EQ',
            'sbi': 'SBIN-EQ',
            'bharti airtel': 'BHARTIARTL-EQ',
            'itc': 'ITC-EQ',
            'kotak': 'KOTAKBANK-EQ',
            'l&t': 'LT-EQ',
            'larsen': 'LT-EQ'
        }
        
        # Sector keywords
        self.sectors = {
            'banking': ['bank', 'hdfc', 'icici', 'sbi', 'kotak', 'axis'],
            'it': ['tcs', 'infosys', 'wipro', 'hcl', 'tech mahindra'],
            'oil': ['reliance', 'ongc', 'oil', 'petroleum', 'gas'],
            'auto': ['tata motors', 'maruti', 'mahindra', 'bajaj', 'hero'],
            'pharma': ['sun pharma', 'dr reddy', 'cipla', 'lupin'],
            'fmcg': ['hindustan unilever', 'itc', 'nestle', 'britannia']
        }
        
        logger.info("✅ News Sentiment Analyzer initialized")
    
    def fetch_indian_market_news(self, hours: int = 4) -> List[NewsArticle]:
        """
        Fetch news from Indian market sources
        
        Args:
            hours: Fetch news from last N hours
        """
        articles = []
        
        # Try multiple sources
        articles.extend(self._fetch_from_newsapi(hours))
        articles.extend(self._fetch_from_moneycontrol())
        articles.extend(self._fetch_from_economic_times())
        
        # Sort by recency and relevance
        articles.sort(key=lambda x: (x.published_at, x.relevance), reverse=True)
        
        logger.info(f"📰 Fetched {len(articles)} news articles")
        return articles[:50]  # Return top 50
    
    def _fetch_from_newsapi(self, hours: int) -> List[NewsArticle]:
        """Fetch from NewsAPI"""
        articles = []
        
        if 'newsapi' not in self.api_keys:
            return articles
        
        try:
            from_time = datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(hours=hours)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.api_keys['newsapi'],
                'q': 'indian stock market OR nifty OR sensex OR bse OR nse',
                'from': from_time.isoformat(),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 30
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('articles', []):
                    news_article = self._parse_article(
                        title=article.get('title', ''),
                        summary=article.get('description', ''),
                        url=article.get('url', ''),
                        source=article.get('source', {}).get('name', 'NewsAPI'),
                        published_at=article.get('publishedAt', '')
                    )
                    
                    if news_article:
                        articles.append(news_article)
        
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
        
        return articles
    
    def _fetch_from_moneycontrol(self) -> List[NewsArticle]:
        """Fetch from MoneyControl (scraping - simplified)"""
        articles = []
        
        try:
            # This is a placeholder - implement proper scraping or use their API
            # For production, use official APIs or RSS feeds
            
            url = "https://www.moneycontrol.com/rss/MCtopnews.xml"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Parse RSS feed (simplified)
                # In production, use feedparser library
                logger.debug("MoneyControl RSS fetched")
        
        except Exception as e:
            logger.debug(f"MoneyControl fetch error: {e}")
        
        return articles
    
    def _fetch_from_economic_times(self) -> List[NewsArticle]:
        """Fetch from Economic Times (scraping - simplified)"""
        articles = []
        
        try:
            # Placeholder - implement proper scraping or API
            url = "https://economictimes.indiatimes.com/rssfeedstopstories.cms"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                logger.debug("Economic Times RSS fetched")
        
        except Exception as e:
            logger.debug(f"Economic Times fetch error: {e}")
        
        return articles
    
    def _parse_article(self, title: str, summary: str, url: str, 
                      source: str, published_at: str) -> Optional[NewsArticle]:
        """Parse and create NewsArticle object"""
        if not title:
            return None
        
        try:
            # Parse published time
            if isinstance(published_at, str):
                pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                pub_time = pub_time.astimezone(ZoneInfo("Asia/Kolkata"))
            else:
                pub_time = datetime.now(ZoneInfo("Asia/Kolkata"))
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(title + ' ' + summary)
            
            # Extract related symbols
            symbols = self._extract_symbols(title + ' ' + summary)
            
            # Determine category
            category = self._categorize_news(title + ' ' + summary)
            
            # Calculate relevance
            relevance = self._calculate_relevance(title, summary, symbols)
            
            return NewsArticle(
                title=title,
                summary=summary or '',
                url=url,
                source=source,
                published_at=pub_time,
                symbols=symbols,
                sentiment_score=sentiment,
                relevance=relevance,
                category=category
            )
        
        except Exception as e:
            logger.error(f"Article parse error: {e}")
            return None
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text
        Returns: -1 (very negative) to 1 (very positive)
        """
        text_lower = text.lower()
        
        # Count positive and negative keywords
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0  # Neutral
        
        # Calculate sentiment score
        total = positive_count + negative_count
        sentiment = (positive_count - negative_count) / total
        
        return sentiment
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract related stock symbols from text"""
        text_lower = text.lower()
        symbols = []
        
        for keyword, symbol in self.symbol_mappings.items():
            if keyword in text_lower:
                symbols.append(symbol)
        
        return list(set(symbols))
    
    def _categorize_news(self, text: str) -> str:
        """Categorize news into market/stock/sector/policy"""
        text_lower = text.lower()
        
        # Check for market-level keywords
        market_keywords = ['nifty', 'sensex', 'market', 'bse', 'nse', 'index']
        if any(kw in text_lower for kw in market_keywords):
            return 'market'
        
        # Check for policy keywords
        policy_keywords = ['rbi', 'sebi', 'government', 'policy', 'budget', 'tax']
        if any(kw in text_lower for kw in policy_keywords):
            return 'policy'
        
        # Check for sector
        for sector, keywords in self.sectors.items():
            if any(kw in text_lower for kw in keywords):
                return f'sector_{sector}'
        
        # Default to stock-specific
        return 'stock'
    
    def _calculate_relevance(self, title: str, summary: str, symbols: List[str]) -> float:
        """Calculate relevance score (0-1)"""
        relevance = 0.5  # Base relevance
        
        # Boost for specific symbols
        if len(symbols) > 0:
            relevance += 0.3
        
        # Boost for strong sentiment keywords in title
        title_lower = title.lower()
        if any(kw in title_lower for kw in self.positive_keywords | self.negative_keywords):
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def get_index_sentiment(self, articles: List[NewsArticle]) -> Dict[str, float]:
        """
        Get overall sentiment for indices and market
        Returns: Dict with 'nifty', 'banknifty', 'market' sentiment scores
        """
        # Filter market-level news
        market_articles = [a for a in articles if a.category == 'market']
        
        if not market_articles:
            return {'nifty': 0.0, 'banknifty': 0.0, 'market': 0.0}
        
        # Calculate weighted average sentiment
        total_weight = sum(a.relevance for a in market_articles)
        
        if total_weight == 0:
            return {'nifty': 0.0, 'banknifty': 0.0, 'market': 0.0}
        
        weighted_sentiment = sum(
            a.sentiment_score * a.relevance for a in market_articles
        ) / total_weight
        
        return {
            'nifty': weighted_sentiment,
            'banknifty': weighted_sentiment,
            'market': weighted_sentiment
        }
    
    def get_stock_sentiment(self, symbol: str, articles: List[NewsArticle]) -> float:
        """
        Get sentiment for specific stock
        Returns: Sentiment score (-1 to 1)
        """
        # Filter articles related to this symbol
        relevant_articles = [a for a in articles if symbol in a.symbols]
        
        if not relevant_articles:
            # Use sector sentiment as fallback
            return self._get_sector_sentiment(symbol, articles)
        
        # Calculate weighted average
        total_weight = sum(a.relevance for a in relevant_articles)
        
        if total_weight == 0:
            return 0.0
        
        sentiment = sum(
            a.sentiment_score * a.relevance for a in relevant_articles
        ) / total_weight
        
        return sentiment
    
    def _get_sector_sentiment(self, symbol: str, articles: List[NewsArticle]) -> float:
        """Get sentiment from sector news"""
        # Determine sector for symbol
        symbol_lower = symbol.lower()
        
        for sector, keywords in self.sectors.items():
            if any(kw in symbol_lower for kw in keywords):
                sector_articles = [
                    a for a in articles 
                    if a.category == f'sector_{sector}'
                ]
                
                if sector_articles:
                    total_weight = sum(a.relevance for a in sector_articles)
                    if total_weight > 0:
                        return sum(
                            a.sentiment_score * a.relevance 
                            for a in sector_articles
                        ) / total_weight
        
        # Fallback to market sentiment
        market_sentiment = self.get_index_sentiment(articles)
        return market_sentiment.get('market', 0.0)
    
    def get_top_movers_news(self, articles: List[NewsArticle]) -> Dict[str, List[NewsArticle]]:
        """
        Get news for stocks with significant sentiment
        Returns: Dict with 'bullish' and 'bearish' lists
        """
        # Group articles by symbol
        symbol_articles = {}
        for article in articles:
            for symbol in article.symbols:
                if symbol not in symbol_articles:
                    symbol_articles[symbol] = []
                symbol_articles[symbol].append(article)
        
        # Calculate sentiment for each symbol
        symbol_sentiments = {}
        for symbol, arts in symbol_articles.items():
            sentiment = self.get_stock_sentiment(symbol, arts)
            symbol_sentiments[symbol] = sentiment
        
        # Get top bullish and bearish
        bullish = sorted(
            [(s, sent) for s, sent in symbol_sentiments.items() if sent > 0.3],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        bearish = sorted(
            [(s, sent) for s, sent in symbol_sentiments.items() if sent < -0.3],
            key=lambda x: x[1]
        )[:5]
        
        return {
            'bullish': [{'symbol': s, 'sentiment': sent, 'articles': symbol_articles[s]} 
                       for s, sent in bullish],
            'bearish': [{'symbol': s, 'sentiment': sent, 'articles': symbol_articles[s]} 
                       for s, sent in bearish]
        }
    
    def convert_to_trading_signal(self, sentiment: float) -> float:
        """
        Convert sentiment score (-1 to 1) to trading signal modifier (0 to 1)
        Used to adjust strategy confidence
        
        -1.0 sentiment -> 0.0 signal (very bearish, reduce buy confidence)
        0.0 sentiment -> 0.5 signal (neutral)
        1.0 sentiment -> 1.0 signal (very bullish, boost buy confidence)
        """
        return (sentiment + 1.0) / 2.0


# Usage Example
"""
# Initialize analyzer
analyzer = NewsSentimentAnalyzer(api_keys={'newsapi': 'your_key'})

# Fetch recent news
articles = analyzer.fetch_indian_market_news(hours=4)

# Get market sentiment
market_sentiment = analyzer.get_index_sentiment(articles)
print(f"NIFTY Sentiment: {market_sentiment['nifty']:.2f}")

# Get stock-specific sentiment
reliance_sentiment = analyzer.get_stock_sentiment('RELIANCE-EQ', articles)
sentiment_signal = analyzer.convert_to_trading_signal(reliance_sentiment)

# Use in trading strategy
ensemble = AdaptiveEnsemble()
signal = ensemble.analyze(
    symbol='RELIANCE-EQ',
    candles=candles,
    current_time=current_time,
    regime=regime,
    news_sentiment=sentiment_signal  # 0.0 to 1.0
)

# Get top movers
movers = analyzer.get_top_movers_news(articles)
print(f"Top Bullish: {[m['symbol'] for m in movers['bullish']]}")
print(f"Top Bearish: {[m['symbol'] for m in movers['bearish']]}")
"""
