# Virtual Environment Setup Guide

## How to Create a Virtual Environment (Windows)

### Option 1: Using `venv` (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using `virtualenv`
```bash
# Install virtualenv first (if not installed)
pip install virtualenv

# Create virtual environment
virtualenv venv

# Activate virtual environment
venv\Scripts\activate
```

## Activating the Virtual Environment

Once created, activate it with:
```bash
venv\Scripts\activate
```

You'll see `(venv)` at the beginning of your command prompt when it's active.

## Deactivating the Virtual Environment

To deactivate:
```bash
deactivate
```

## Running the Application

After activating the virtual environment and installing dependencies:

```bash
# Activate venv first
venv\Scripts\activate

# Run the FastAPI application
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000
API Documentation: http://localhost:8000/docs

