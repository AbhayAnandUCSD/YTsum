# YouTube Video Summarizer

A multimodal YouTube video summarizer that combines audio, transcript, and visual content to generate comprehensive summaries using OpenAI's models.

## Features

- Downloads YouTube videos and extracts audio
- Generates transcripts using OpenAI's Whisper
- Samples and analyzes keyframes from the video
- Combines transcript and visual content
- Generates comprehensive summaries using GPT-4
- Supports both long-form and timestamped summaries
- Optional Streamlit web interface

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/YTsum.git
cd YTsum
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface
```bash
python src/main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --output_format long
```

### Web Interface
```bash
streamlit run src/app.py
```

## Project Structure

```
YTsum/
├── src/
│   ├── main.py           # CLI entry point
│   ├── app.py            # Streamlit web interface
│   ├── video/
│   │   ├── downloader.py # Video download utilities
│   │   └── processor.py  # Video processing utilities
│   ├── audio/
│   │   └── transcriber.py # Audio transcription
│   ├── vision/
│   │   └── analyzer.py   # Visual content analysis
│   └── summarizer/
│       └── generator.py  # Summary generation
├── requirements.txt
└── README.md
```

## License

MIT
