#!/bin/bash

# SETUP THE ENVIRONMENT

# Create a Python virtual environment if it doesn't already exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate the virtual environment
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi

# Install required Python packages
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install required packages"
        exit 1
    fi
else
    echo "requirements.txt not found. Skipping installation."
fi
