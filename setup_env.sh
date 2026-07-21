#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Creating a new virtual environment 'venv'..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Virtual environment setup is complete!"
echo "To activate it, run: source venv/bin/activate"
