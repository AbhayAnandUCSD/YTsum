import streamlit as st
import os
from video.downloader import VideoDownloader
import tempfile

st.set_page_config(page_title="YouTube Video Downloader", page_icon="🎥")

st.title("YouTube Video Downloader")
st.write("Download videos and audio from YouTube URLs")

# Create a temporary directory for downloads
temp_dir = os.path.join(tempfile.gettempdir(), "youtube_downloads")
os.makedirs(temp_dir, exist_ok=True)

# Initialize the downloader
downloader = VideoDownloader(output_dir=temp_dir)

# Input for YouTube URL
url = st.text_input("Enter YouTube URL:")

if st.button("Download"):
    if url:
        try:
            with st.spinner("Downloading video and audio..."):
                video_path, audio_path = downloader.download_video(url)
                
                # Display success message
                st.success("Download completed!")
                
                # Show file paths
                st.write(f"Video saved to: {video_path}")
                st.write(f"Audio saved to: {audio_path}")
                
                # Add download buttons
                with open(video_path, 'rb') as video_file:
                    st.download_button(
                        label="Download Video",
                        data=video_file,
                        file_name=os.path.basename(video_path),
                        mime="video/mp4"
                    )
                
                with open(audio_path, 'rb') as audio_file:
                    st.download_button(
                        label="Download Audio",
                        data=audio_file,
                        file_name=os.path.basename(audio_path),
                        mime="audio/mp3"
                    )
                
                # Cleanup files after download
                downloader.cleanup(video_path, audio_path)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a YouTube URL") 