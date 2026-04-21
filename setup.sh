#!/usr/bin/env bash
# =============================================================================
# setup.sh  —  Create virtual environment and install all dependencies
# Usage:  bash setup.sh
# =============================================================================
set -e

echo "======================================================"
echo " Neuromorphic ADBScan — Project Setup"
echo " Patent: US 10,510,154 (Intel / Rita Chattopadhyay)"
echo "======================================================"

# Check Python 3.10+
PY=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PY"

# Create virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment already exists — skipping creation."
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate
echo "Activated: $(which python3)"

# Upgrade pip
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================================"
echo " Setup complete."
echo " To activate: source venv/bin/activate"
echo " Then run:    python run_benchmark.py"
echo "======================================================"
