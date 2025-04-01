#!/usr/bin/env python
# AI-Automated Content Video Pipeline - Fully Automated Version
# This script runs the entire pipeline automatically without user input

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the environment for the pipeline"""
    print("\n\033[1m🔧 Setting up environment...\033[0m")
    
    # Create output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    output_dir.joinpath("images").mkdir(exist_ok=True)
    
    # Set Gemini API key from config file if available
    try:
        from config import GEMINI_API_KEY
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        print("✅ Loaded Gemini API key from config.py")
    except ImportError:
        # Check if Gemini API key is set in environment
        if "GEMINI_API_KEY" not in os.environ:
            # For Colab, try to get from userdata
            try:
                from google.colab import userdata
                api_key = userdata.get("GEMINI_API_KEY")
                if api_key:
                    os.environ["GEMINI_API_KEY"] = api_key
                    print("✅ Retrieved Gemini API key from Colab userdata")
                else:
                    raise ValueError("No API key found in userdata")
            except:
                # Fallback to environment variable or error
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    print("\n\033[91m❌ ERROR: GEMINI_API_KEY environment variable is not set.\033[0m")
                    print("Please set your Gemini API key as an environment variable:")
                    print("  - In Linux/macOS: export GEMINI_API_KEY=your-api-key")
                    print("  - In Windows: set GEMINI_API_KEY=your-api-key")
                    print("  - In Colab: Set the userdata or environment variable")
                    print("  - Or create a config.py file with GEMINI_API_KEY variable")
                    sys.exit(1)
    
    # Install required packages
    print("\n\033[1m📦 Installing dependencies...\033[0m")
    packages = [
        "google-ai-generativelanguage",
        "google-api-core",
        "numpy",
        "torch",
        "scipy",
        "pillow",
        "imageio",
        "imageio-ffmpeg"
    ]
    
    for package in packages:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.SubprocessError:
            print(f"\n\033[93m⚠️ Warning: Could not install {package}\033[0m")
    
    # Install Bark
    try:
        import bark
    except ImportError:
        print("\n\033[1m📦 Installing Bark...\033[0m")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/suno-ai/bark.git"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✅ FFmpeg is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("\n\033[93m⚠️ FFmpeg not found. Attempting to install...\033[0m")
        try:
            # For Google Colab
            subprocess.run(
                ["apt-get", "update", "-qq", "&&", "apt-get", "install", "-y", "-qq", "ffmpeg"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            print("✅ FFmpeg installed successfully")
        except:
            print("\n\033[91m❌ Could not install FFmpeg. Manual installation required:\033[0m")
            print("  - Windows: Use chocolatey or download from https://ffmpeg.org/download.html")
            print("  - Linux: sudo apt-get install ffmpeg")
            print("  - macOS: brew install ffmpeg")
    
    # Clone Open Sora if not already present
    if not Path("./Open-Sora-Plan-v1.0.0-hf").exists():
        print("\n\033[1m📥 Cloning Open Sora repository...\033[0m")
        try:
            subprocess.run(
                ["git", "clone", "-b", "dev", "https://github.com/camenduru/Open-Sora-Plan-v1.0.0-hf"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Install Open Sora dependencies
            print("\n\033[1m📦 Installing Open Sora dependencies...\033[0m")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "diffusers==0.24.0", "gradio==3.50.2", 
                 "einops==0.7.0", "omegaconf==2.1.1", "pytorch-lightning==1.4.2", 
                 "torchmetrics==0.6.0", "torchtext==0.6", "accelerate==0.28.0"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.SubprocessError as e:
            print(f"\n\033[91m❌ Failed to clone Open Sora or install dependencies: {e}\033[0m")
            sys.exit(1)
    
    print("\n\033[1m✅ Environment setup complete!\033[0m")
    return output_dir

def run_pipeline():
    """Run the AI Content Pipeline automatically"""
    print("\n\033[1m🚀 Starting Automated AI Content Pipeline...\033[0m\n")
    
    # Setup environment
    setup_environment()
    
    # Default creative prompt
    default_prompt = (
        "Create a visual story about a futuristic city where AI assists humans in every aspect of life. "
        "The story follows a day in the life of a young professional named Maya who interacts with various "
        "AI systems throughout her day. Include vivid descriptions of the city's architecture, Maya's "
        "smart home, autonomous vehicles, personal AI assistants, and AR/VR interfaces. Show both the "
        "benefits and challenges of living in this advanced society. Include dialogue between Maya and "
        "her AI systems, as well as interactions with other humans."
    )
    
    # Run the pipeline with the default prompt and options
    cmd = [
        sys.executable, 
        "ai_content_pipeline.py",
        "--prompt", default_prompt,
        "--video_steps", "50",
        "--video_scale", "10.0",
        "--voice_preset", "v2/en_speaker_1",
        "--resolution", "1080p",
        "--fps", "30",
        "--use_images"
    ]
    
    print(f"\n\033[1m🤖 Running pipeline with default prompt:\033[0m")
    print(f"\033[93m\"{default_prompt}\"\033[0m\n")
    
    try:
        # Execute the pipeline with default parameters
        # We use communicate() to capture output while still showing it to the user
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Show output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        # Wait for the process to complete
        process.wait()
        
        if process.returncode != 0:
            print(f"\n\033[91m❌ Pipeline execution failed with return code {process.returncode}\033[0m")
            return False
            
    except Exception as e:
        print(f"\n\033[91m❌ Error during pipeline execution: {e}\033[0m")
        return False
    
    print("\n\033[1m✅ Automated pipeline completed successfully!\033[0m")
    
    # Find the generated final video
    try:
        videos = list(Path("./output").glob("final_video_*.mp4"))
        if videos:
            latest_video = max(videos, key=lambda p: p.stat().st_mtime)
            print(f"\n\033[1m🎥 Final video created at: \033[92m{latest_video}\033[0m")
            
            # Display different instructions based on environment
            try:
                import google.colab
                print("\nTo view the video in Colab, run this code in a new cell:")
                print("```python")
                print("from IPython.display import HTML")
                print("from base64 import b64encode")
                print(f"\nmp4_path = '{latest_video}'")
                print("mp4 = open(mp4_path,'rb').read()")
                print("data_url = \"data:video/mp4;base64,\" + b64encode(mp4).decode()")
                print("\nHTML(\"\"\"")
                print("<video width=640 controls>")
                print("      <source src=\"%s\" type=\"video/mp4\">")
                print("</video>")
                print("\"\"\" % data_url)")
                print("```")
            except ImportError:
                # Not in Colab, provide local instructions
                print("\nTo view the video, open it with your default video player.")
        else:
            print("\n\033[93m⚠️ No final video file found in the output directory.\033[0m")
    except Exception as e:
        print(f"\n\033[93m⚠️ Could not locate final video: {e}\033[0m")
    
    return True

if __name__ == "__main__":
    run_pipeline()
