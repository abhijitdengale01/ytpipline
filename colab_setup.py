# AI Content Pipeline - Colab Setup Script
# This script handles setup and execution in Google Colab environments

import os
import sys
import subprocess
import shutil

# Clear any existing cloned repositories to avoid nested directories
def clear_existing_repo():
    print("📦 Cleaning up existing directories...")
    if os.path.exists('ytpipline'):
        shutil.rmtree('ytpipline')

# Clone the GitHub repository
def clone_repository():
    print("📥 Cloning repository...")
    subprocess.run(["git", "clone", "https://github.com/abhijitdengale01/ytpipline.git"], check=True)
    os.chdir('ytpipline')

# Install dependencies with specific versions
def install_dependencies():
    print("📦 Installing dependencies...")
    subprocess.run(["pip", "install", "huggingface_hub==0.16.4"], check=True)
    subprocess.run(["pip", "install", "diffusers==0.19.3"], check=True)
    subprocess.run(["pip", "install", "google-generativeai"], check=True)
    subprocess.run(["pip", "install", "torch torchvision"], check=True)
    subprocess.run(["pip", "install", "transformers"], check=True)
    subprocess.run(["pip", "install", "imageio imageio-ffmpeg"], check=True)
    subprocess.run(["pip", "install", "scipy numpy"], check=True)

# Set environment variables
def setup_environment(api_key):
    print("🔑 Setting up API key...")
    os.environ["GEMINI_API_KEY"] = api_key

# Run the pipeline
def run_pipeline():
    print("🚀 Running pipeline...")
    subprocess.run(["python", "run_automated_pipeline.py"], check=True)

def main():
    print("\n🌟 Setting up AI Content Pipeline in Colab 🌟\n")
    
    # Get API key from user
    api_key = input("Please enter your Gemini API key: ")
    if not api_key:
        print("⚠️ API key is required!")
        return
    
    try:
        clear_existing_repo()
        clone_repository()
        install_dependencies()
        setup_environment(api_key)
        run_pipeline()
        print("\n✅ Pipeline execution complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
