import os
import urllib.parse

# -------------------------------
# API Keys and Credentials Config
# -------------------------------

# Polygon API Key
POLYGON_API_KEY = "V8EQF2XleGLSXoBXk_h1A7bxiowxjb8Q"

# Financial Prep API Key
FINANCIAL_PREP_API_KEY = "cLLHJ7ThMUnzrBqQDJCXaz25mZzKsqYL"

# MongoDB Atlas Credentials
MONGO_DB_USER = "shadowguy311"  # your username (use your email or configured username)
MONGO_DB_PASS = "tanki12"      # your password

# Encode username and password per RFC 3986
encoded_user = urllib.parse.quote_plus(MONGO_DB_USER)
encoded_pass = urllib.parse.quote_plus(MONGO_DB_PASS)

# MongoDB connection string (Atlas)
mongo_url = (
    f"mongodb+srv://{encoded_user}:{encoded_pass}@cluster0.yjmg8.mongodb.net/"
    f"?retryWrites=true&w=majority&appName=Cluster0"
)

# Alpaca Paper Trading API Credentials
ALPACA_API_KEY = "PKAALI77U1Z60VUCJX2O"
ALPACA_API_SECRET = "eP5XJlfSpb9GaRwQNFfngxmuFi8EKODmRYazRPDu"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Finnhub API Credentials
FINNHUB_API_KEY = "cv1vs2hr01qngf0b5390cv1vs2hr01qngf0b539g"
FINNHUB_SECRET = "cv1vs2hr01qngf0b53ag"  # if needed for your integration

# NewsAPI Credentials
NEWSAPI_KEY = "027e167533f7488bb9935e9ab1874e72"

# Alpha Vantage API Credentials
ALPHA_VANTAGE_API_KEY = "1A4W4OWOQ1KR2J5A"
