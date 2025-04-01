# Run this directly in Colab to set up and run the AI Content Pipeline

import os
import sys
import subprocess
import shutil

# Clear existing repo
print("\n🔄 Clearing existing repository...")
if os.path.exists('ytpipline'):
    shutil.rmtree('ytpipline')

# Clone repository
print("\n🔄 Cloning repository...")
subprocess.run(["git", "clone", "https://github.com/abhijitdengale01/ytpipline.git"], check=True)
os.chdir('ytpipline')

# Install dependencies
print("\n🔄 Installing dependencies...")
subprocess.run(["pip", "install", "-q", "huggingface_hub==0.16.4"], check=True)
subprocess.run(["pip", "install", "-q", "diffusers==0.19.3"], check=True)
subprocess.run(["pip", "install", "-q", "google-generativeai"], check=True)
subprocess.run(["pip", "install", "-q", "torch"], check=True)
subprocess.run(["pip", "install", "-q", "torchvision"], check=True)
subprocess.run(["pip", "install", "-q", "transformers"], check=True)
subprocess.run(["pip", "install", "-q", "imageio", "imageio-ffmpeg"], check=True)
subprocess.run(["pip", "install", "-q", "scipy", "numpy"], check=True)

# Get API key from user
api_key = input("\n✅ Dependencies installed. Please enter your Gemini API key: ")
if not api_key:
    print("⚠️ API key is required!")
    sys.exit(1)

# Set environment variable
os.environ["GEMINI_API_KEY"] = api_key

# Run pipeline
print("\n🚀 Running pipeline...")
subprocess.run(["python", "run_automated_pipeline.py"], check=True)

print("\n✅ Pipeline execution complete!")
