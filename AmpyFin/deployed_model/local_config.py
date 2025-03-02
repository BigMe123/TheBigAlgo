# config.py - Standalone configuration file
# Place this in your deployed_model directory
import urllib.parse

# API keys
POLYGON_API_KEY = "V8EQF2XleGLSXoBXk_h1A7bxiowxjb8Q"
FINANCIAL_PREP_API_KEY = "cLLHJ7ThMUnzrBqQDJCXaz25mZzKsqYL"
MONGO_DB_USER = "shadowguy311"
MONGO_DB_PASS = "tanki12" 
API_KEY = "PKAALI77U1Z60VUCJX2O"
API_SECRET = "eP5XJlfSpb9GaRwQNFfngxmuFi8EKODmRYazRPDu"
BASE_URL = "https://paper-api.alpaca.markets"

# Encode credentials for MongoDB connection string
encoded_user = urllib.parse.quote_plus(MONGO_DB_USER)
encoded_pass = urllib.parse.quote_plus(MONGO_DB_PASS)

# MongoDB connection string
mongo_url = f"mongodb+srv://{encoded_user}:{encoded_pass}@cluster0.yjmg8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"