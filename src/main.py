import argparse
import os
from video.downloader import VideoDownloader
from video.processor import VideoProcessor
from audio.transcriber import AudioTranscriber
from vision.analyzer import VisionAnalyzer
from summarizer.generator import SummaryGenerator
import tempfile

def main():
    parser = argparse.ArgumentParser(description="YouTube Video Summarizer")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--output_format",
        choices=["long", "timestamped"],
        default="long",
        help="Summary output format"
    )
    parser.add_argument(
        "--output_file",
        help="Output file path for the summary (optional)"
    )
    args = parser.parse_args()

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Initialize components
            downloader = VideoDownloader(temp_dir)
            processor = VideoProcessor(interval_seconds=5)
            transcriber = AudioTranscriber()
            analyzer = VisionAnalyzer()
            generator = SummaryGenerator()

            print("Downloading video...")
            video_path, audio_path = downloader.download_video(args.url)

            print("Extracting keyframes...")
            keyframes = processor.extract_keyframes(video_path)

            print("Transcribing audio...")
            transcript_segments = transcriber.transcribe_audio(audio_path)
            transcript = transcriber.format_transcript(
                transcript_segments,
                format_type=args.output_format
            )

            print("Analyzing visual content...")
            visual_analyses = analyzer.analyze_keyframes(keyframes)
            visual_content = analyzer.format_visual_content(visual_analyses)

            print("Generating summary...")
            summary = generator.generate_summary(
                transcript=transcript,
                visual_content=visual_content,
                format_type=args.output_format
            )

            # Output the summary
            if args.output_file:
                with open(args.output_file, "w") as f:
                    f.write(summary)
                print(f"\nSummary saved to: {args.output_file}")
            else:
                print("\nGenerated Summary:")
                print("=" * 80)
                print(summary)
                print("=" * 80)

        except Exception as e:
            print(f"Error: {str(e)}")
            return 1

    return 0

if __name__ == "__main__":
    exit(main()) 