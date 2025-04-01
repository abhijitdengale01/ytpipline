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

# Gemini imports
from google import genai
from google.genai import types

# Bark imports
from bark import SAMPLE_RATE, generate_audio, preload_models
import scipy.io.wavfile as wavfile

# Open Sora imports will be dynamically loaded to avoid import errors
# if the repo is not yet cloned

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
        subprocess.run(
            ["git", "clone", "-b", "dev", "https://github.com/camenduru/Open-Sora-Plan-v1.0.0-hf"],
            check=True
        )
        
        # Install dependencies for Open Sora
        print("\n🔄 Installing Open Sora dependencies...")
        subprocess.run(
            ["pip", "install", "-q", "diffusers==0.24.0", "gradio==3.50.2", "einops==0.7.0", 
             "omegaconf==2.1.1", "pytorch-lightning==1.4.2", "torchmetrics==0.6.0", 
             "torchtext==0.6", "accelerate==0.28.0"],
            check=True
        )
    
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
                types.Part.from_text(text="""{}}""".format(prompt)),
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

def generate_video_with_open_sora(prompt=None, image_paths=None, sample_steps=50, scale=10.0, seed=0, randomize_seed=True):
    """Generate video using Open Sora
    
    Args:
        prompt (str, optional): The prompt to generate video from
        image_paths (list, optional): List of paths to images to use as input
        sample_steps (int): Number of sampling steps
        scale (float): Guidance scale
        seed (int): Random seed
        randomize_seed (bool): Whether to randomize the seed
        
    Returns:
        str: Path to the generated video
    """
    # Either prompt or image_paths must be provided
    if prompt is None and (image_paths is None or len(image_paths) == 0):
        raise ValueError("Either prompt or image_paths must be provided")
        
    print("\n🎨 Generating video with Open Sora...")
    
    # Change to Open Sora directory
    prev_dir = os.getcwd()
    os.chdir("./Open-Sora-Plan-v1.0.0-hf")
    
    # Dynamically import Open Sora components
    import sys
    sys.path.append("./")
    
    import torch
    from diffusers import PNDMScheduler
    from transformers import T5Tokenizer, T5EncoderModel
    from PIL import Image
    
    from opensora.models.ae import ae_stride_config, getae, getae_wrapper
    from opensora.models.diffusion.latte.modeling_latte import LatteT2V
    from opensora.sample.pipeline_videogen import VideoGenPipeline
    
    # Setup args
    class Args:
        def __init__(self):
            self.ae = 'CausalVAEModel_4x8x8'
            self.force_images = False
            self.model_path = 'LanguageBind/Open-Sora-Plan-v1.0.0'
            self.text_encoder_name = 'DeepFloyd/t5-v1_1-xxl'
            self.version = '65x512x512'
    
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n📥 Loading Open Sora models (using {device})...")
    
    # Set random seed if randomizing
    if randomize_seed:
        seed = random.randint(0, 203279)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Load model
    transformer_model = LatteT2V.from_pretrained(
        args.model_path, 
        subfolder=args.version, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        cache_dir='cache_dir'
    ).to(device)

    vae = getae_wrapper(args.ae)(
        args.model_path, 
        subfolder="vae", 
        cache_dir='cache_dir'
    ).to(device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    
    vae.vae.enable_tiling()
    image_size = int(args.version.split('x')[1])
    latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
    vae.latent_size = latent_size
    transformer_model.force_images = args.force_images
    
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir="cache_dir")
    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder_name, 
        cache_dir="cache_dir",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    # Set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()
    scheduler = PNDMScheduler()
    videogen_pipeline = VideoGenPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer_model
    ).to(device=device)
    
    print("\n🎬 Generating video...")
    # Generate the video
    video_length = transformer_model.config.video_length
    height, width = int(args.version.split('x')[1]), int(args.version.split('x')[2])
    num_frames = int(args.version.split('x')[0])
    
    # If image_paths is provided, we'll use a different approach to create a video from images
    if image_paths and len(image_paths) > 0:
        # Create a video from the images
        print(f"Creating video from {len(image_paths)} images...")
        
        # Load and resize images to match required dimensions
        frames = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                img = img.resize((width, height))
                frames.append(np.array(img))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # If no valid images, fall back to generating from prompt
        if len(frames) == 0 and prompt:
            print("No valid images found, falling back to prompt-based generation")
            # Use the normal generation method as below
            with torch.no_grad():
                videos = videogen_pipeline(
                    prompt,
                    video_length=video_length,
                    height=height,
                    width=width,
                    num_inference_steps=sample_steps,
                    guidance_scale=scale,
                    enable_temporal_attentions=True,
                    num_images_per_prompt=1,
                    mask_feature=True,
                ).video
                videos = videos[0]
        else:
            # Create video from frames
            videos = np.array(frames)
    else:
        # Generate the video using the prompt
        with torch.no_grad():
            videos = videogen_pipeline(
                prompt,
                video_length=video_length,
                height=height,
                width=width,
                num_inference_steps=sample_steps,
                guidance_scale=scale,
                enable_temporal_attentions=True,
                num_images_per_prompt=1,
                mask_feature=True,
            ).video
            videos = videos[0]

    # Save the video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"../output/generated_video_{timestamp}.mp4"
    imageio.mimwrite(video_path, videos, fps=24, quality=9)  # highest quality is 10, lowest is 0
    
    # Return to previous directory
    os.chdir(prev_dir)
    
    print(f"\n✅ Video generation complete!")
    print(f"📹 Video saved to {video_path}")
    print(f"Video info: {videos.shape[0]} frames at {height}x{width}, Steps: {sample_steps}, Scale: {scale}, Seed: {seed}")
    
    return video_path

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
    
    print(f"Generating audio for {len(text_segments)} text segments...")
    
    # Generate audio for each segment
    audio_arrays = []
    for i, segment in enumerate(text_segments):
        print(f"\nGenerating segment {i+1}/{len(text_segments)}...")
        print(f"Text: {segment[:50]}..." if len(segment) > 50 else f"Text: {segment}")
        audio_array = generate_audio(segment, history_prompt=voice_preset)
        audio_arrays.append(audio_array)
    
    # Combine all audio segments
    combined_audio = np.concatenate(audio_arrays)
    
    # Save the audio file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = f"./output/generated_audio_{timestamp}.wav"
    wavfile.write(audio_path, SAMPLE_RATE, combined_audio)
    
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
            sample_steps=args.video_steps,
            scale=args.video_scale,
            seed=args.video_seed,
            randomize_seed=True
        )
    else:
        video_path = generate_video_with_open_sora(
            prompt=args.prompt,
            sample_steps=args.video_steps,
            scale=args.video_scale,
            seed=args.video_seed,
            randomize_seed=True
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
