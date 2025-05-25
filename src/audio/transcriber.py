import openai
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AudioTranscriber:
    def __init__(self):
        """Initialize the audio transcriber with OpenAI API key."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        openai.api_key = self.api_key

    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """Transcribe audio file using OpenAI's Whisper model.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            List[Dict]: List of transcription segments with timestamps
        """
        try:
            with open(audio_path, "rb") as audio_file:
                # Transcribe audio with timestamps
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
                
                # Extract segments with timestamps
                segments = []
                for segment in response.segments:
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
                
                return segments
                
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")

    def format_transcript(self, segments: List[Dict], format_type: str = "long") -> str:
        """Format transcript based on desired output format.
        
        Args:
            segments (List[Dict]): List of transcription segments
            format_type (str): Output format type ("long" or "timestamped")
            
        Returns:
            str: Formatted transcript
        """
        if format_type == "long":
            return " ".join(segment["text"] for segment in segments)
        else:  # timestamped
            formatted = []
            for segment in segments:
                start_time = self._format_timestamp(segment["start"])
                formatted.append(f"[{start_time}] {segment['text']}")
            return "\n".join(formatted)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS format.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" 