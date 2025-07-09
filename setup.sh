#!/bin/bash

echo "Setting up Face Overlay App..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete."
echo "To activate the environment next time, run: source venv/bin/activate"
echo "Then run the app with: python main.py"
