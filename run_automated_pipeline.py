#!/usr/bin/env python
# AI-Automated Content Video Pipeline - Fully Automated Version
# This script runs the entire pipeline automatically without user input

import os
import sys
import subprocess
from pathlib import Path
import datetime
import random
import torch
import imageio
from diffusers import PNDMScheduler
from transformers import T5Tokenizer, T5EncoderModel

# Note: Do not import opensora modules here as they won't be available until after setup_environment runs

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
            # Use the dev branch as specified
            subprocess.run(
                ["git", "clone", "-b", "dev", "https://github.com/camenduru/Open-Sora-Plan-v1.0.0-hf"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.SubprocessError:
            # If that fails, try without the branch specification
            try:
                print("\n\033[93m⚠️ Dev branch failed, trying main branch...\033[0m")
                subprocess.run(
                    ["git", "clone", "https://github.com/camenduru/Open-Sora-Plan-v1.0.0-hf"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.SubprocessError:
                # If that fails, try alternative repository sources
                try:
                    print("\n\033[93m⚠️ Main repository failed, trying alternative source...\033[0m")
                    subprocess.run(
                        ["git", "clone", "https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0", "Open-Sora-Plan-v1.0.0-hf"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                except subprocess.SubprocessError as e:
                    try:
                        # Create the directory manually if all cloning attempts fail
                        print("\n\033[93m⚠️ Repository cloning failed, creating directory structure manually...\033[0m")
                        os.makedirs("./Open-Sora-Plan-v1.0.0-hf", exist_ok=True)
                        os.makedirs("./Open-Sora-Plan-v1.0.0-hf/opensora", exist_ok=True)
                        
                        # Create minimal required files
                        Path("./Open-Sora-Plan-v1.0.0-hf/opensora/__init__.py").touch()
                        
                        print("\n\033[93m⚠️ Manual directory creation complete. The pipeline will use Gemini-generated images only.\033[0m")
                    except Exception as e2:
                        print(f"\n\033[91m❌ Failed to set up Open Sora structure: {e2}\033[0m")
                        sys.exit(1)
                    
        try:
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
            print(f"\n\033[93m⚠️ Failed to install some dependencies: {e}\033[0m")
            print("\n\033[93m⚠️ The pipeline will continue but may not have full functionality.\033[0m")
    
    print("\n\033[1m✅ Environment setup complete!\033[0m")
    return output_dir

def generate_video_with_open_sora(prompt=None, image_paths=None, video_steps=50, video_scale=10.0, video_seed=0):
    """Generate a video using Open Sora from a text prompt or images"""
    print("\n\033[1m📹 Generating video with Open Sora...\033[0m")
    
    if prompt is None and (image_paths is None or len(image_paths) == 0):
        raise ValueError("Either prompt or image_paths must be provided")
    
    # Generate a timestamp for the output filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/generated_video_{timestamp}.mp4"
    
    try:
        # Setup the Open Sora model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Args configuration
        args = type('args', (), {
            'ae': 'CausalVAEModel_4x8x8',
            'force_images': image_paths is not None and len(image_paths) > 0,
            'model_path': 'LanguageBind/Open-Sora-Plan-v1.0.0',
            'text_encoder_name': 'DeepFloyd/t5-v1_1-xxl',
            'version': '65x512x512'
        })
        
        # Set environment seed
        seed = video_seed if video_seed > 0 else random.randint(0, 203279)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Import Open Sora components here, after the repo has been cloned
        sys.path.append("./Open-Sora-Plan-v1.0.0-hf")
        
        # Now import the OpenSora modules
        print("\n\033[1mImporting OpenSora modules...\033[0m")
        from opensora.models.ae import ae_stride_config, getae, getae_wrapper
        from opensora.models.diffusion.latte.modeling_latte import LatteT2V
        from opensora.sample.pipeline_videogen import VideoGenPipeline
        
        print("\n\033[1mLoading transformer model...\033[0m")
        transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, 
                                                   torch_dtype=torch.float16, cache_dir='cache_dir').to(device)

        print("\n\033[1mLoading VAE model...\033[0m")
        vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir').to(device, dtype=torch.float16)
        vae.vae.enable_tiling()
        image_size = int(args.version.split('x')[1])
        latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
        vae.latent_size = latent_size
        transformer_model.force_images = args.force_images
        
        print("\n\033[1mLoading text encoder and tokenizer...\033[0m")
        tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir="cache_dir")
        text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir="cache_dir",
                                                    torch_dtype=torch.float16).to(device)

        # Set all models to eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        
        # Setup scheduler and pipeline
        scheduler = PNDMScheduler()
        videogen_pipeline = VideoGenPipeline(vae=vae,
                                           text_encoder=text_encoder,
                                           tokenizer=tokenizer,
                                           scheduler=scheduler,
                                           transformer=transformer_model).to(device=device)

        # Generate the video
        with torch.no_grad():
            print(f"\n\033[1mGenerating video with prompt: {prompt}\033[0m")
            video_length = transformer_model.config.video_length if not args.force_images else 1
            height, width = int(args.version.split('x')[1]), int(args.version.split('x')[2])
            num_frames = 1 if video_length == 1 else int(args.version.split('x')[0])
            
            videos = videogen_pipeline(prompt,
                                     video_length=video_length,
                                     height=height,
                                     width=width,
                                     num_inference_steps=int(video_steps),
                                     guidance_scale=float(video_scale),
                                     enable_temporal_attentions=not args.force_images,
                                     num_images_per_prompt=1,
                                     mask_feature=True).video

        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Save the video
        videos = videos[0]
        imageio.mimwrite(output_path, videos, fps=24, quality=9)  # highest quality is 10, lowest is 0
        print(f"\n\033[1m📹 Video saved to {output_path}\033[0m")
        
        # Display model info
        display_model_info = f"Video size: {num_frames}×{height}×{width}, Sampling Step: {video_steps}, Guidance Scale: {video_scale}"
        print(f"\n\033[1m{display_model_info}\033[0m")
        
        return output_path
    
    except Exception as e:
        print(f"\n\033[91m🚫 Error generating video with Open Sora: {str(e)}\033[0m")
        print("\n\033[93m📝 Falling back to using Gemini-generated images only...\033[0m")
        
        # If we have images but Open Sora failed, create a simple slideshow from the images
        if image_paths and len(image_paths) > 0:
            try:
                print("\n\033[1m📸 Creating slideshow from Gemini-generated images...\033[0m")
                # Create slideshow from images (5 seconds per image)
                images = []
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        img = imageio.imread(img_path)
                        # Duplicate each image to create a 5-second clip (assuming 30fps)
                        for _ in range(150):  # 5 seconds at 30fps
                            images.append(img)
                
                # Save as video
                if images:
                    imageio.mimwrite(output_path, images, fps=30, quality=9)
                    print(f"\n\033[1m📹 Slideshow video saved to {output_path}\033[0m")
                    return output_path
            except Exception as e2:
                print(f"\n\033[91m🚫 Error creating slideshow: {str(e2)}\033[0m")
        
        # If all else fails, raise the original exception
        raise e

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
