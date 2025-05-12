#!/bin/bash

set -e  # ถ้ามี error ใดๆ จะหยุดทันที

echo "📦 STEP 1: Compress and Upload model to S3 + Log to MLflow..."
python scripts/compress/compressAndUpload.py

echo ""
echo "🚀 STEP 2: Deploy to SageMaker endpoint..."
python scripts/deploy/sagemaker_deploy.py

echo ""
echo "✅ All steps completed successfully."
