import os
from typing import Tuple
from pytube import YouTube
import tempfile

class VideoDownloader:
    def __init__(self, output_dir: str = None):
        """Initialize the video downloader.
        
        Args:
            output_dir (str, optional): Directory to save downloaded files. 
                                      If None, uses system temp directory.
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)

    def download_video(self, url: str) -> Tuple[str, str]:
        """Download video and audio from YouTube URL.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            Tuple[str, str]: Paths to downloaded video and audio files
        """
        try:
            # Create YouTube object
            yt = YouTube(url)
            
            # Get video stream (highest resolution)
            video_stream = yt.streams.filter(progressive=True).get_highest_resolution()
            
            # Download video
            video_path = video_stream.download(output_path=self.output_dir)
            
            # Get audio stream and download
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_path = audio_stream.download(output_path=self.output_dir)
            
            # Rename audio file to .mp3
            base, _ = os.path.splitext(audio_path)
            new_audio_path = base + '.mp3'
            os.rename(audio_path, new_audio_path)
            
            return video_path, new_audio_path
            
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")

    def cleanup(self, video_path: str, audio_path: str):
        """Clean up downloaded files.
        
        Args:
            video_path (str): Path to video file
            audio_path (str): Path to audio file
        """
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}") 