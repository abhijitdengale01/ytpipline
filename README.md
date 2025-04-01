# 🌟 AI-Automated Content Video Pipeline

This pipeline automates the creation of content videos by integrating three powerful AI tools:
- **Text & Image Generation**: Gemini for script creation and image generation
- **Visual Storytelling**: Open Sora for video generation from prompts or images
- **Audio Synthesis**: Bark for voice narration

## 📋 Running in Google Colab

To run this pipeline in Google Colab, follow these steps:

1. Create a new Colab notebook
2. Add and run the following cells:

### Step 1: Clone the repository and install dependencies

```python
# Clone the repository 
!git clone https://github.com/YOUR_USERNAME/ai-content-pipeline.git
%cd ai-content-pipeline

# Install requirements
!pip install -r requirements.txt

# Install Bark
!pip install git+https://github.com/suno-ai/bark.git

# Install ffmpeg
!apt-get update && apt-get install -y ffmpeg
```

### Step 2: Set up your Gemini API key

```python
# Set your Gemini API key
import os

# Method 1: Set it directly (not recommended for security reasons)
# os.environ["GEMINI_API_KEY"] = "your-api-key-here"

# Method 2: Upload from a file (more secure)
from google.colab import files
uploaded = files.upload()  # Upload a text file with your key
with open(list(uploaded.keys())[0], 'r') as f:
    os.environ["GEMINI_API_KEY"] = f.read().strip()

# Method 3: Use the Colab secrets (most secure, if available)
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')

# Create an output directory
!mkdir -p output/images
```

### Step 3: Upload or create the pipeline script

```python
# Option 1: Upload your existing script
from google.colab import files
uploaded = files.upload()  # Upload your ai_content_pipeline.py

# Option 2: Or create the script directly in Colab
# !wget https://raw.githubusercontent.com/YOUR_USERNAME/ai-content-pipeline/main/ai_content_pipeline.py
```

### Step 4: Run the pipeline

```python
# Run with a specific prompt
!python ai_content_pipeline.py --prompt "A visual story about a futuristic city where AI assists humans in daily life. The story follows a day in the life of a young professional named Maya who interacts with various AI systems throughout her day."

# Or run interactively
!python ai_content_pipeline.py
```

### Step 5: View and download the results

```python
# Display the generated video
from IPython.display import HTML
from base64 import b64encode

mp4_file = sorted([f for f in os.listdir('output') if f.startswith('final_video') and f.endswith('.mp4')])[-1]
mp4_path = os.path.join('output', mp4_file)

mp4 = open(mp4_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

# Download the results
from google.colab import files
files.download(mp4_path)
```

## 📋 Requirements

- Python 3.8+
- Gemini API Key
- FFmpeg installed on your system
- CUDA-compatible GPU (recommended for faster video generation)

## 🚀 Local Usage

Run the pipeline locally with the following command:

```bash
python ai_content_pipeline.py --prompt "Your creative prompt here"
```

### Optional Arguments

```
--prompt TEXT          Prompt for content generation
--video_steps INT      Number of sampling steps for video generation (default: 50)
--video_scale FLOAT    Guidance scale for video generation (default: 10.0)
--video_seed INT       Seed for video generation (default: 0, randomized if not specified)
--voice_preset STR     Voice preset for Bark (default: "v2/en_speaker_1")
--resolution STR       Output video resolution (default: "1080p")
--fps INT              Output video frame rate (default: 30)
--use_images           Use generated images for video creation
```

## 📝 Example Prompt

```
"Generate a visual story about a futuristic city where AI assists humans in every aspect of life. Include character dialogues and vivid descriptions."
```

## 📂 Output

All generated files are saved in the `./output` directory:
- Text content: `generated_content_[timestamp].txt`
- Generated images: `./output/images/generated_image_[timestamp]_[number].[ext]`
- Video: `generated_video_[timestamp].mp4`
- Audio: `generated_audio_[timestamp].wav`
- Final video: `final_video_[timestamp].mp4`

## 📚 Pipeline Steps

1. **Content & Image Generation** (Gemini) - Creates a story/script and images from your prompt
2. **Visual Creation** (Open Sora) - Transforms the story into a video or uses generated images to create a video
3. **Audio Generation** (Bark) - Creates voice narration from the script
4. **Video & Audio Merging** (FFmpeg) - Synchronizes audio and visuals
5. **Quality Check** - Ensures the video is high quality and properly synced

## ⚠️ Notes

- The first run will take longer as it downloads and sets up the required models
- Video generation is computationally intensive and works best with a good GPU
- For best results, craft detailed and descriptive prompts
- The Gemini image generation model (gemini-2.0-flash-exp-image-generation) will generate both text and images from your prompt
