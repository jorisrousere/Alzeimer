#!/bin/bash

echo "Processing data..."
bash scripts/process_data.sh

echo "Training model..."
bash scripts/train.sh

echo "Testing model..."
bash scripts/test_pipeline.sh
