#!/bin/bash

# Check if lrzuntar is installed
if ! command -v lrzip &> /dev/null
then
    echo "lrzip is not installed. Installing it..."

    # Install lrzuntar
    sudo apt install lrzip

    # Check if install was successful
    if command -v lrzip &> /dev/null
    then
        echo "lrzip installed successfully!"
        
    else 
        echo "Failed to install lrzip!"
        exit 1
    fi
else
    echo "lrzip is already installed." 
fi

echo "Decompressing dataset..."
lrzuntar Dataset/pre-training/CodeChangeNet.jsonl.tar.lrz
mv CodeChangeNet.jsonl Dataset/pre-training/

echo "Preparing pre-training dataset..."
env python src/DataProcessing/prep_CGN_dataset.py

exit 0