import os
from typing import Tuple
from pytube import YouTube
from moviepy.editor import VideoFileClip
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
            
            # Extract audio
            video_clip = VideoFileClip(video_path)
            audio_path = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_audio.mp3")
            video_clip.audio.write_audiofile(audio_path)
            
            # Close the video clip
            video_clip.close()
            
            return video_path, audio_path
            
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