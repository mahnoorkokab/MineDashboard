"""
Vercel Serverless Function Handler for FastAPI
This file handles all API requests and serves the FastAPI application
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import app
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import the FastAPI app
from app import app

# Use Mangum to convert FastAPI (ASGI) to AWS Lambda/Vercel format
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    # Fallback if mangum is not available
    def handler(event, context):
        return {
            "statusCode": 500,
            "body": "Mangum is required. Install with: pip install mangum"
        }

def lambda_handler(event, context):
    """
    Vercel serverless function entry point
    This function is called by Vercel for all API requests
    """
    return handler(event, context)
