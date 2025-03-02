import os
import urllib.parse

# Hardcoded configuration values (replace with your actual credentials)
POLYGON_API_KEY = "V8EQF2XleGLSXoBXk_h1A7bxiowxjb8Q"
FINANCIAL_PREP_API_KEY = "cLLHJ7ThMUnzrBqQDJCXaz25mZzKsqYL"

# Use your email as username if that's what you configured in Atlas:
MONGO_DB_USER = "shadowguy311"  # use your email as username
MONGO_DB_PASS = "tanki12"

API_KEY = "PKAALI77U1Z60VUCJX2O"
API_SECRET = "eP5XJlfSpb9GaRwQNFfngxmuFi8EKODmRYazRPDu"
BASE_URL = "https://paper-api.alpaca.markets"

# Escape username and password according to RFC 3986
encoded_user = urllib.parse.quote_plus(MONGO_DB_USER)
encoded_pass = urllib.parse.quote_plus(MONGO_DB_PASS)

# MongoDB connection string matching the provided link format
mongo_url = f"mongodb+srv://{encoded_user}:{encoded_pass}@cluster0.yjmg8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"