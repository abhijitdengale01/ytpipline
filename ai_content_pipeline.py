# AI-Automated Content Video Pipeline
# This script integrates Gemini, Open Sora, and Bark to create automated content videos

import os
import json
import torch
import numpy as np
import argparse
import subprocess
import imageio
import tempfile
import mimetypes
import random
from datetime import datetime
from pathlib import Path
import sys

# Gemini imports
from google import genai
from google.genai import types

# Bark imports
from bark import SAMPLE_RATE, generate_audio, preload_models
import scipy.io.wavfile as wavfile

def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()

def setup_environment():
    """Set up the environment for all components of the pipeline"""
    print("\n📦 Setting up the environment...")
    
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
        # Check for Gemini API key in environment
        if "GEMINI_API_KEY" not in os.environ:
            api_key = input("Enter your Gemini API key: ")
            os.environ["GEMINI_API_KEY"] = api_key
    
    # Check if Open Sora repo exists, if not clone it
    if not Path("./Open-Sora-Plan-v1.0.0-hf").exists():
        print("\n🔄 Cloning Open Sora repository...")
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
                print("\n⚠️ Dev branch failed, trying main branch...")
                subprocess.run(
                    ["git", "clone", "https://github.com/camenduru/Open-Sora-Plan-v1.0.0-hf"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.SubprocessError:
                # If that fails, try alternative repository sources
                try:
                    print("\n⚠️ Main repository failed, trying alternative source...")
                    subprocess.run(
                        ["git", "clone", "https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0", "Open-Sora-Plan-v1.0.0-hf"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                except subprocess.SubprocessError as e:
                    try:
                        # Create the directory manually if all cloning attempts fail
                        print("\n⚠️ Repository cloning failed, creating directory structure manually...")
                        os.makedirs("./Open-Sora-Plan-v1.0.0-hf", exist_ok=True)
                        os.makedirs("./Open-Sora-Plan-v1.0.0-hf/opensora", exist_ok=True)
                        
                        # Create minimal required files
                        Path("./Open-Sora-Plan-v1.0.0-hf/opensora/__init__.py").touch()
                        
                        print("\n⚠️ Manual directory creation complete. The pipeline will use Gemini-generated images only.")
                    except Exception as e2:
                        print(f"\n❌ Failed to set up Open Sora structure: {e2}")
                        sys.exit(1)
                    
        try:
            # Install Open Sora dependencies
            print("\n🔄 Installing Open Sora dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "diffusers==0.24.0", "gradio==3.50.2", 
                 "einops==0.7.0", "omegaconf==2.1.1", "pytorch-lightning==1.4.2", 
                 "torchmetrics==0.6.0", "torchtext==0.6", "accelerate==0.28.0"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.SubprocessError as e:
            print(f"\n⚠️ Failed to install some dependencies: {e}")
            print("\n⚠️ The pipeline will continue but may not have full functionality.")
    
    # Install Bark if not already installed
    try:
        import bark
    except ImportError:
        print("\n🔄 Installing Bark...")
        subprocess.run(
            ["pip", "install", "git+https://github.com/suno-ai/bark.git"],
            check=True
        )
    
    # Install ffmpeg if not already installed
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("\n⚠️ ffmpeg not found. Please install ffmpeg manually.")
        print("On Windows: Use chocolatey or download from https://ffmpeg.org/download.html")
        print("On Linux: sudo apt-get install ffmpeg")
        print("On macOS: brew install ffmpeg")
    
    print("\n✅ Environment setup complete!")
    return output_dir

def generate_content_with_gemini(prompt):
    """Generate content and images using Gemini API
    
    Args:
        prompt (str): The prompt to generate content from
        
    Returns:
        tuple: The generated content, content file path, and list of generated image paths
    """
    print("\n📝 Generating content and images with Gemini...")
    
    # Use the exact API structure as provided
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash-exp-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="{}".format(prompt)),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "image",
            "text",
        ],
        response_mime_type="text/plain",
    )

    full_response = ""
    image_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            # Save images to the images folder
            img_counter = len(image_files) + 1
            file_name = f"./output/images/generated_image_{timestamp}_{img_counter}"
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            full_path = f"{file_name}{file_extension}"
            
            save_binary_file(full_path, inline_data.data)
            image_files.append(full_path)
            
            print(f"Image of type {inline_data.mime_type} saved to: {full_path}")
        else:
            # This is text content
            print(chunk.text, end="")
            if chunk.text:
                full_response += chunk.text
    
    print("\n\n✅ Content and image generation complete!")
    
    # Save the generated text content to a file
    content_file = f"./output/generated_content_{timestamp}.txt"
    with open(content_file, "w", encoding="utf-8") as f:
        f.write(full_response)
    
    print(f"📄 Content saved to {content_file}")
    print(f"🖼️ Generated {len(image_files)} images.")
    
    return full_response, content_file, image_files

def generate_video_with_open_sora(prompt=None, image_paths=None, video_steps=50, video_scale=10.0, video_seed=0):
    """Generate a video using Open Sora from a text prompt or images"""
    print("\n🎬 Generating video with Open Sora...")
    
    if prompt is None and (image_paths is None or len(image_paths) == 0):
        raise ValueError("Either prompt or image_paths must be provided")
    
    # Generate a timestamp for the output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        print("Importing OpenSora modules...")
        from opensora.models.ae import ae_stride_config, getae, getae_wrapper
        from opensora.models.diffusion.latte.modeling_latte import LatteT2V
        from opensora.sample.pipeline_videogen import VideoGenPipeline

        # Load model components
        from diffusers import PNDMScheduler
        from transformers import T5Tokenizer, T5EncoderModel
        
        print("Loading transformer model...")
        transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, 
                                                   torch_dtype=torch.float16, cache_dir='cache_dir').to(device)

        print("Loading VAE model...")
        vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir').to(device, dtype=torch.float16)
        vae.vae.enable_tiling()
        image_size = int(args.version.split('x')[1])
        latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
        vae.latent_size = latent_size
        transformer_model.force_images = args.force_images
        
        print("Loading text encoder and tokenizer...")
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
            print(f"Generating video with prompt: {prompt}")
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
        print(f"Video saved to {output_path}")
        
        # Display model info
        display_model_info = f"Video size: {num_frames}×{height}×{width}, Sampling Step: {video_steps}, Guidance Scale: {video_scale}"
        print(display_model_info)
        
        return output_path
    
    except Exception as e:
        print(f"\n❌ Error generating video with Open Sora: {str(e)}")
        print("Falling back to using Gemini-generated images only...")
        
        # If we have images but Open Sora failed, create a simple slideshow from the images
        if image_paths and len(image_paths) > 0:
            try:
                print("Creating slideshow from Gemini-generated images...")
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
                    print(f"Slideshow video saved to {output_path}")
                    return output_path
            except Exception as e2:
                print(f"\n❌ Error creating slideshow: {str(e2)}")
        
        # If all else fails, raise the original exception
        raise e

def generate_audio_with_bark(text, voice_preset="v2/en_speaker_1"):
    """Generate audio using Bark
    
    Args:
        text (str): The text to convert to speech
        voice_preset (str): The voice preset to use
        
    Returns:
        str: Path to the generated audio file
    """
    print("\n🔊 Generating audio with Bark...")
    
    # Preload Bark models
    print("📥 Loading Bark models...")
    from bark import SAMPLE_RATE, generate_audio, preload_models
    preload_models()
    
    # Check if text is too long and split if necessary
    max_length = 250  # Bark works best with shorter segments
    text_segments = []
    
    if len(text) > max_length:
        # Simple split by sentences
        sentences = text.replace('\n', ' ').split('.')
        current_segment = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_segment) + len(sentence) < max_length:
                current_segment += sentence + "."
            else:
                if current_segment:
                    text_segments.append(current_segment)
                current_segment = sentence + "."
        
        if current_segment:  # Add the last segment
            text_segments.append(current_segment)
    else:
        text_segments = [text]
    
    print(f"Split text into {len(text_segments)} segments for processing")
    
    # Generate timestamp for output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = f"./output/generated_audio_{timestamp}.wav"
    
    # Create directories if they don't exist
    os.makedirs("./output/temp", exist_ok=True)
    
    if len(text_segments) > 1:
        # Process each segment separately
        segment_audios = []
        for i, segment in enumerate(text_segments):
            print(f"Processing segment {i+1}/{len(text_segments)}...")
            audio_array = generate_audio(segment, history_prompt=voice_preset)
            segment_path = f"./output/temp/segment_{i}_{timestamp}.wav"
            from scipy.io.wavfile import write as write_wav
            write_wav(segment_path, SAMPLE_RATE, audio_array)
            segment_audios.append(segment_path)
        
        # Combine audio segments using ffmpeg
        import subprocess
        # Create a file list for ffmpeg
        with open(f"./output/temp/filelist_{timestamp}.txt", "w") as f:
            for audio_file in segment_audios:
                f.write(f"file '{os.path.abspath(audio_file)}'\n")
        
        # Concatenate audio files
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", 
            "-i", f"./output/temp/filelist_{timestamp}.txt", 
            "-c", "copy", audio_path
        ], check=True)
    else:
        # Process single segment
        audio_array = generate_audio(text_segments[0], history_prompt=voice_preset)
        from scipy.io.wavfile import write as write_wav
        write_wav(audio_path, SAMPLE_RATE, audio_array)
    
    print(f"\n✅ Audio generation complete!")
    print(f"🔉 Audio saved to {audio_path}")
    
    return audio_path

def merge_video_audio(video_path, audio_path, output_path=None, target_resolution="1080p", fps=30):
    """Merge video and audio using ffmpeg
    
    Args:
        video_path (str): Path to the video file
        audio_path (str): Path to the audio file
        output_path (str, optional): Path to the output file. Defaults to None.
        target_resolution (str, optional): Target resolution. Defaults to "1080p".
        fps (int, optional): Frame rate. Defaults to 30.
        
    Returns:
        str: Path to the merged video
    """
    print("\n🎬 Merging video and audio...")
    
    # Create output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./output/final_video_{timestamp}.mp4"
    
    # Determine target resolution
    if target_resolution == "1080p":
        resolution = "1920:1080"
    elif target_resolution == "720p":
        resolution = "1280:720"
    elif target_resolution == "480p":
        resolution = "854:480"
    else:
        resolution = "1920:1080"  # Default to 1080p
    
    # Run ffmpeg to merge video and audio
    try:
        # First, get video duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
             "stream=width,height,avg_frame_rate,codec_name", "-of", 
             "json=compact=1", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        video_info = json.loads(result.stdout)["streams"][0]
        
        # Get audio duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", 
             "stream=codec_name,sample_rate,channels", "-of", 
             "json=compact=1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        audio_info = json.loads(result.stdout)["streams"][0]
        
        print(f"Video duration: {video_info['duration']:.2f}s, Audio duration: {audio_info['duration']:.2f}s")
        
        # If audio is longer than video, loop the video to match audio duration
        if float(audio_info['duration']) > float(video_info['duration']):
            print(f"Audio ({audio_info['duration']:.2f}s) is longer than video ({video_info['duration']:.2f}s). Looping video...")
            temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            loop_count = int(np.ceil(float(audio_info['duration']) / float(video_info['duration'])))
            
            # Create a text file with the list of videos to concatenate
            concat_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
            for _ in range(loop_count):
                concat_file.write(f"file '{os.path.abspath(video_path)}'\n")
            concat_file.close()
            
            # Concatenate the videos
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file.name, 
                 "-c", "copy", temp_video],
                check=True
            )
            
            # Trim the concatenated video to match the audio duration
            input_video = temp_video
            os.remove(concat_file.name)
        else:
            input_video = video_path
        
        # Merge video and audio, and scale to target resolution
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_video, "-i", audio_path, "-c:v", "libx264", 
             "-preset", "slow", "-crf", "18", "-c:a", "aac", "-b:a", "192k", 
             "-shortest", "-vf", f"scale={resolution}", "-r", str(fps), output_path],
            check=True
        )
        
        # Clean up temporary files if created
        if float(audio_info['duration']) > float(video_info['duration']) and os.path.exists(temp_video):
            os.remove(temp_video)
            
    except subprocess.SubprocessError as e:
        print(f"❌ Error merging video and audio: {e}")
        return None
    
    print(f"\n✅ Video and audio merging complete!")
    print(f"🎥 Final video saved to {output_path}")
    
    return output_path

def quality_check(final_video_path):
    """Perform a quality check on the final video
    
    Args:
        final_video_path (str): Path to the final video
    """
    print("\n🔍 Performing quality check...")
    
    try:
        # Get video information
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
             "stream=width,height,avg_frame_rate,codec_name", "-of", 
             "json=compact=1", final_video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        video_info = json.loads(result.stdout)["streams"][0]
        
        # Get audio information
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", 
             "stream=codec_name,sample_rate,channels", "-of", 
             "json=compact=1", final_video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        audio_info = json.loads(result.stdout)["streams"][0]
        
        # Display information
        print("\nVideo Quality Check Results:")
        print(f"📊 Resolution: {video_info['width']}x{video_info['height']}")
        print(f"🎞️ Frame Rate: {video_info['avg_frame_rate']}")
        print(f"🎬 Video Codec: {video_info['codec_name']}")
        print(f"🔊 Audio Codec: {audio_info['codec_name']}")
        print(f"📢 Audio Sample Rate: {audio_info['sample_rate']} Hz")
        print(f"🎧 Audio Channels: {audio_info['channels']}")
        
        print("\n✅ Quality check complete!")
        print("\n🎬 To view the video, you can:")
        print(f"1. Open it directly at: {os.path.abspath(final_video_path)}")
        print("2. Use a media player like VLC, Windows Media Player, or QuickTime.")
        
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error during quality check: {e}")

def main():
    """Main function to run the AI-Automated Content Video Pipeline"""
    print("\n🚀 Starting AI-Automated Content Video Pipeline...")
    
    # Setup environment
    output_dir = setup_environment()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI-Automated Content Video Pipeline")
    parser.add_argument("--prompt", type=str, help="Prompt for content generation")
    parser.add_argument("--video_steps", type=int, default=50, help="Video generation sample steps")
    parser.add_argument("--video_scale", type=float, default=10.0, help="Video generation guidance scale")
    parser.add_argument("--video_seed", type=int, default=0, help="Video generation seed")
    parser.add_argument("--voice_preset", type=str, default="v2/en_speaker_1", help="Voice preset for Bark")
    parser.add_argument("--resolution", type=str, default="1080p", help="Output video resolution")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    parser.add_argument("--use_images", action="store_true", help="Use generated images for video creation")
    
    args = parser.parse_args()
    
    # Get user prompt if not provided
    if not args.prompt:
        print("\n💡 Example Prompt: 'Generate a visual story about a futuristic city where AI assists humans in every aspect of life. Include character dialogues and vivid descriptions.'")
        args.prompt = input("\n🔤 Enter your prompt for content generation: ")
    
    # Step 1: Generate content with Gemini
    content, content_file, image_files = generate_content_with_gemini(args.prompt)
    
    # Ask user if they want to continue with the generated content
    proceed = input("\n⏭️ Continue with this generated content? (y/n): ").lower()
    if proceed != 'y':
        print("\n❌ Pipeline stopped by user.")
        return
    
    # Ask user if they want to use the generated images for video creation
    use_images = args.use_images
    if len(image_files) > 0 and not args.use_images:
        use_images = input(f"\n📸 Use the {len(image_files)} generated images to create the video? (y/n): ").lower() == 'y'
    
    # Step 2: Generate video with Open Sora
    if use_images and len(image_files) > 0:
        video_path = generate_video_with_open_sora(
            image_paths=image_files,
            video_steps=args.video_steps,
            video_scale=args.video_scale,
            video_seed=args.video_seed,
        )
    else:
        video_path = generate_video_with_open_sora(
            prompt=args.prompt,
            video_steps=args.video_steps,
            video_scale=args.video_scale,
            video_seed=args.video_seed,
        )
    
    # Step 3: Generate audio with Bark
    # Ask user if they want to modify the content for audio generation
    print("\n📝 For audio generation, you can:")
    print("1. Use the entire generated content")
    print("2. Use just the dialogue portions")
    print("3. Enter a custom narration prompt")
    
    audio_choice = input("\n🔢 Enter your choice (1/2/3): ")
    
    audio_text = content
    if audio_choice == "2":
        # Extract dialogue (simple implementation - can be improved)
        dialogue_lines = [line for line in content.split('\n') if ':' in line or line.strip().startswith('"') or line.strip().startswith("'")]
        audio_text = '\n'.join(dialogue_lines)
        print(f"\n📜 Extracted {len(dialogue_lines)} lines of dialogue.")
    elif audio_choice == "3":
        print("\n💡 Example: 'Hello, I am the AI narrator of this futuristic city story. Follow me as I guide you through the lives of people living alongside intelligent machines.'")
        audio_text = input("\n🔤 Enter your custom narration text: ")
    
    audio_path = generate_audio_with_bark(audio_text, voice_preset=args.voice_preset)
    
    # Step 4: Merge video and audio
    final_video_path = merge_video_audio(
        video_path=video_path,
        audio_path=audio_path,
        target_resolution=args.resolution,
        fps=args.fps
    )
    
    # Step 5: Quality check
    if final_video_path:
        quality_check(final_video_path)
    
    print("\n🎉 AI-Automated Content Video Pipeline completed successfully!")
    print(f"📁 All outputs are saved in the {output_dir} directory.")

if __name__ == "__main__":
    main()
