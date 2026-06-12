# FinSense ML Backend

FastAPI backend for **FinSense**, a real-time financial analysis and fair value prediction platform. The service collects market data, serves stock chart and news APIs, stores financial records in MongoDB, and combines machine learning signals for user-facing prediction responses.

## What This Backend Does

- Tracks a configured list of major US stocks using Finnhub quote data.
- Stores real-time and historical market records in MongoDB Atlas.
- Serves paginated financial news with cache refresh support.
- Provides chart data for multiple time ranges.
- Builds prediction responses from recent market patterns, time-series forecasts, and recent news sentiment.
- Uses LSTM-related market-pattern logic, TimesFM forecasting, FinBERT sentiment analysis, and calibrated signal weighting.
- Returns explanation text and public signal names suitable for a frontend application.

## Tech Stack

- **API:** FastAPI, Uvicorn
- **Language:** Python 3.12+
- **Database:** MongoDB Atlas, Motor, PyMongo
- **Market Data:** Finnhub, Yahoo Finance / yfinance
- **ML / Data:** PyTorch, TimesFM, Transformers, scikit-learn, NumPy, Pandas
- **Testing:** Python `unittest`
- **Package / Runtime:** uv

## API Routes

### News

```http
GET /api/news?page=1&limit=10
```

Returns cached or refreshed financial news.

Response includes:

- `status`
- `page`
- `limit`
- `total`
- `totalPages`
- `refreshing`
- `data`

### Stock Chart Data

```http
GET /api/stocks/{ticker}/chart?period=1m
```

Supported chart periods:

```txt
1d, 1w, 1m, 3m, 1y, 2y, 3y
```

For periods longer than `1d`, the API may append the latest real-time quote to the historical series.

### Prediction

```http
GET /api/predict/{ticker}?period=1w&explain=true
```

Supported prediction periods:

```txt
1d, 1w, 1m, 3m, 6m, 1y
```

Prediction responses include:

- `status`
- `ticker`
- `period`
- `data_points`
- `latest_price`
- `latest_price_time`
- `signal_sources`
- `prediction`
- `weights`
- `signals`
- `explanation`
- `explanatory_text`

Recent news sentiment is used for short-horizon predictions (`1d`, `1w`) and documented as unused for longer horizons.

### Admin Latest Inserts

```http
GET /api/admin/latest?limit=20
```

Returns the latest real-time stock records from the database.

## Tracked Stocks

The real-time polling service currently tracks:

```txt
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, NFLX, INTC,
CSCO, ADBE, QCOM, PYPL, AMD, TMUS, COST, BKNG, AMGN, SBUX
```

## Environment Variables

Create a `.env` file in the project root.

```env
MONGO_URI=your_mongodb_connection_string
# or
MONGODB_URI=your_mongodb_connection_string

FINNHUB_API_KEY=your_finnhub_news_api_key
FINNHUB_API_KEY2=your_finnhub_stock_quote_api_key

ENABLE_POLLER=false

NEWS_REFRESH_INTERVAL_SECONDS=3600
NEWS_LOOKBACK_DAYS=7
NEWS_INITIAL_REFRESH_DELAY_SECONDS=15
NEWS_MAX_CONCURRENT_REQUESTS=10

USE_FINBERT_FOR_NEWS=false
USE_FINBERT_FOR_PREDICTION=true
MAX_PREDICTION_SENTIMENT_ARTICLES=5

ENABLE_INVESTOR_AGENT=true
ENABLE_INVESTOR_AGENT_V2=true
INVESTOR_AGENT_CHECK_INTERVAL_SECONDS=3600
INVESTOR_AGENT_INITIAL_DELAY_SECONDS=30
```

For local development, `ENABLE_POLLER=false` is recommended unless you intentionally want the background quote poller to run.

## Running Locally

Install dependencies with uv:

```bash
uv sync
```

Start the FastAPI app:

```bash
uv run uvicorn main:app --reload
```

The API will be available at:

```txt
http://127.0.0.1:8000
```

FastAPI docs:

```txt
http://127.0.0.1:8000/docs
```

## Running Tests

Run the unittest suite:

```bash
uv run python -m unittest discover -v
```

The current test coverage includes:

- News endpoint cache, pagination, and refresh behavior
- News relevance filtering
- Lightweight and FinBERT sentiment mapping
- LSTM inference frame preparation
- Prediction response shape and explanation behavior
- Short-horizon and long-horizon sentiment rules
- Weight adjustment and calibrated signal behavior

## Paper-Trading Investor Agents

The investor agents are isolated from the prediction models and existing market data collections. They use the same `MONGODB_URI` and `stock_tracking_db`, but write only to agent-specific collections.

Version 1 is the original 8-day run and keeps its initial behavior:

```txt
agent_runs
agent_decisions
agent_trades
agent_portfolio_snapshots
```

Start a live 8-day paper-trading run:

```http
POST /api/agent/start
```

Starting the agent immediately runs the first decision cycle. While the API process is alive, a background scheduler checks for an active run and executes at most one decision cycle per calendar day. At the end of the 8-day window, the next scheduler check liquidates every open position automatically. You can also liquidate manually:

```http
POST /api/agent/liquidate
```

View results:

```http
GET /api/agent/status
GET /api/agent/history
```

Version 2 is a separate 5-day run. It can top up existing high-confidence positions while still respecting the max position cap:

```txt
agent_v2_runs
agent_v2_decisions
agent_v2_trades
agent_v2_portfolio_snapshots
```

Start the separate v2 run:

```http
POST /api/agent-v2/start
```

View v2 results:

```http
GET /api/agent-v2/status
GET /api/agent-v2/history
```

Disable automatic agent operation after the tests by setting:

```env
ENABLE_INVESTOR_AGENT=false
ENABLE_INVESTOR_AGENT_V2=false
```

## Project Structure

```txt
main.py                         FastAPI app, routes, background poller, news cache
data/fetcher.py                 Financial news retrieval and filtering pipeline
src/services/prediction_service.py
                                Prediction response orchestration
src/models/lstm_inference.py    LSTM inference helpers
src/models/timesfm_inference.py TimesFM forecast helpers
src/models/nlp_sentiment.py     FinBERT sentiment analysis
src/models/weight_adjustor.py   Signal weighting and explanation logic
tests/                          Backend unit tests
saved_models/                   Saved model artifacts and calibration data
data/local_storage/             Local news cache
```

## Purpose

FinSense is designed to support real-time financial decision analysis by combining:

- Current stock quotes
- Historical market behavior
- S&P 500 benchmark context
- Financial news sentiment
- Machine learning prediction signals
- User-facing explanations

The backend focuses on reliability, clear API contracts, and prediction output that can be safely presented by a frontend dashboard.
