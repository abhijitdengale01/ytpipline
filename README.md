# 🌟 AI-Automated Content Video Pipeline

This pipeline automates the creation of content videos by integrating three powerful AI tools:
- **Text & Image Generation**: Gemini for script creation and image generation
- **Visual Storytelling**: Open Sora for video generation from prompts or images
- **Audio Synthesis**: Bark for voice narration

## 📋 One-Click Automated Pipeline

The simplest way to use this pipeline is with the fully automated script:

```bash
python run_automated_pipeline.py
```

This script handles everything automatically:
- Sets up the environment and dependencies
- Uses a default creative prompt
- Generates content, video, and audio
- Merges everything into a final video

## 📋 Running in Google Colab

### Method 1: Quick Setup (Recommended)

1. Create a new Google Colab notebook
2. Run this single cell to set up everything automatically:

```python
# Clone the repository
!git clone https://github.com/abhijitdengale01/ytpipline.git
%cd ytpipline

# Set your Gemini API key
%env GEMINI_API_KEY=your_api_key_here  # Replace with your actual API key

# Run the automated pipeline
!python run_automated_pipeline.py
```

### Method 2: Step-by-Step Setup

1. **Clone the repository**
   ```python
   !git clone https://github.com/abhijitdengale01/ytpipline.git
   %cd ytpipline
   ```

2. **Create the config.py file with your API key**
   ```python
   %%writefile config.py
   # Configuration file for AI Content Pipeline
   GEMINI_API_KEY = "your_api_key_here"  # Replace with your actual API key
   ```

3. **Set up the required directories**
   ```python
   !mkdir -p output/images
   ```

4. **Run the pipeline**
   ```python
   !python run_automated_pipeline.py
   ```

5. **View the generated video**
   ```python
   from IPython.display import HTML
   from base64 import b64encode
   import os
   
   # Find the latest generated video
   video_files = [f for f in os.listdir('output') if f.startswith('final_video') and f.endswith('.mp4')]
   if video_files:
       latest_video = max(video_files, key=lambda x: os.path.getctime(os.path.join('output', x)))
       video_path = os.path.join('output', latest_video)
       
       # Display the video
       mp4 = open(video_path, 'rb').read()
       data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
       
       display(HTML(f"""
       <video width=640 controls>
             <source src="{data_url}" type="video/mp4">
       </video>
       """))
   else:
       print("No video files found in the output directory.")
   ```

## 📋 Custom Prompts and Options

If you want to customize the pipeline:

```python
# Use a custom prompt
!python ai_content_pipeline.py --prompt "Your custom prompt here"

# With additional options
!python ai_content_pipeline.py --prompt "Your custom prompt" --video_steps 100 --video_scale 15.0 --use_images
```

## 🚨 Troubleshooting in Colab

### Memory Issues
If you encounter memory errors in Colab:

1. Use a GPU or TPU runtime
   - Go to Runtime > Change runtime type > Select GPU

2. Restart the runtime if it becomes sluggish
   - Runtime > Restart runtime

### API Key Issues
If you have problems with the Gemini API key:

1. Make sure you're using a valid API key
2. Use the config.py method instead of environment variables
3. Check if your API key has the proper permissions for image generation

## 📁 All Pipeline Files

- **run_automated_pipeline.py**: One-click automated solution
- **ai_content_pipeline.py**: Main pipeline implementation
- **config.py**: API key configuration (you create this)
- **requirements.txt**: Required Python packages

## 📝 Pipeline Steps

1. **Content & Image Generation** (Gemini) - Creates a story/script and images from your prompt
2. **Visual Creation** (Open Sora) - Transforms the story into a video or uses generated images to create a video
3. **Audio Generation** (Bark) - Creates voice narration from the script
4. **Video & Audio Merging** (FFmpeg) - Synchronizes audio and visuals
5. **Quality Check** - Ensures the video is high quality and properly synced

## 📋 Requirements

- Python 3.8+
- Gemini API Key
- FFmpeg (installed automatically in Colab)
- CUDA-compatible GPU (recommended for faster video generation)

## ⚠️ Notes

- The first run will take longer as it downloads and sets up the required models
- Video generation is computationally intensive and works best with a good GPU
- For best results, craft detailed and descriptive prompts
- The Gemini image generation model (gemini-2.0-flash-exp-image-generation) will generate both text and images from your prompt
