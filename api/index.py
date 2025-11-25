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

# Change to parent directory so file paths work correctly
os.chdir(parent_dir)

# Import the FastAPI app
from app import app

# Use Mangum to convert FastAPI (ASGI) to AWS Lambda/Vercel format
from mangum import Mangum

# Create the handler - this is what Vercel will call
handler = Mangum(app, lifespan="off")

# Export the handler for Vercel
__all__ = ["handler"]
