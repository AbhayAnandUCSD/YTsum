import streamlit as st
import tempfile
import os
from video.downloader import VideoDownloader
from video.processor import VideoProcessor
from audio.transcriber import AudioTranscriber
from vision.analyzer import VisionAnalyzer
from summarizer.generator import SummaryGenerator

def main():
    st.title("YouTube Video Summarizer")
    st.write("Generate comprehensive summaries of YouTube videos using AI")

    # Input
    url = st.text_input("Enter YouTube URL:")
    output_format = st.radio(
        "Summary Format",
        ["long", "timestamped"],
        format_func=lambda x: "Long-form Summary" if x == "long" else "Timestamped Summary"
    )

    if st.button("Generate Summary"):
        if not url:
            st.error("Please enter a YouTube URL")
            return

        with st.spinner("Processing video..."):
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Initialize components
                    downloader = VideoDownloader(temp_dir)
                    processor = VideoProcessor(interval_seconds=5)
                    transcriber = AudioTranscriber()
                    analyzer = VisionAnalyzer()
                    generator = SummaryGenerator()

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Download video
                    status_text.text("Downloading video...")
                    video_path, audio_path = downloader.download_video(url)
                    progress_bar.progress(20)

                    # Extract keyframes
                    status_text.text("Extracting keyframes...")
                    keyframes = processor.extract_keyframes(video_path)
                    progress_bar.progress(40)

                    # Transcribe audio
                    status_text.text("Transcribing audio...")
                    transcript_segments = transcriber.transcribe_audio(audio_path)
                    transcript = transcriber.format_transcript(
                        transcript_segments,
                        format_type=output_format
                    )
                    progress_bar.progress(60)

                    # Analyze visual content
                    status_text.text("Analyzing visual content...")
                    visual_analyses = analyzer.analyze_keyframes(keyframes)
                    visual_content = analyzer.format_visual_content(visual_analyses)
                    progress_bar.progress(80)

                    # Generate summary
                    status_text.text("Generating summary...")
                    summary = generator.generate_summary(
                        transcript=transcript,
                        visual_content=visual_content,
                        format_type=output_format
                    )
                    progress_bar.progress(100)

                    # Display results
                    st.success("Summary generated successfully!")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Summary", "Transcript", "Visual Analysis"])
                    
                    with tab1:
                        st.markdown(summary)
                    
                    with tab2:
                        st.text(transcript)
                    
                    with tab3:
                        st.text(visual_content)

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 