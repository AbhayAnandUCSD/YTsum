import streamlit as st
import os
import tempfile
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎬 YouTube Video Summarizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform YouTube videos into comprehensive summaries using AI</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_choice = st.selectbox(
        "GPT Model",
        ["gpt-4o-mini"],
        help="Choose which GPT model to use for summarization"
    )
    
    # Output format selection
    output_format = st.selectbox(
        "Summary Format",
        ["long", "timestamped"],
        help="Long: Continuous summary | Timestamped: Summary with timestamps"
    )
    
    # Processing interval
    interval_seconds = st.slider(
        "Keyframe Interval (seconds)",
        min_value=1,
        max_value=10,
        value=5,
        help="How often to extract keyframes for visual analysis"
    )
    
    # Max tokens for summary
    max_tokens = st.slider(
        "Max Summary Length",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum tokens for the generated summary"
    )
    
    # API key check
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found in environment variables")
        st.info("Please set your OpenAI API key in a .env file or environment variable")
    else:
        st.success(f"✅ OpenAI API key configured")
        st.info(f"Using model: {model_choice}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 Video Input")
    
    # URL input
    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter the YouTube video URL you want to summarize"
    )
    
    # Process button
    process_button = st.button(
        "🚀 Generate Summary",
        type="primary",
        disabled=not url or not api_key,
        help="Click to start the summarization process"
    )

with col2:
    st.header("ℹ️ How it works")
    st.markdown("""
    1. **Download** video and extract audio (Mock)
    2. **Transcribe** audio using OpenAI Whisper (Mock)
    3. **Analyze** visual content using CLIP (Mock)
    4. **Generate** comprehensive summary using GPT-4 (Real API)
    """)

# Processing section
if process_button and url and api_key:
    st.markdown("---")
    st.header("🔄 Processing Pipeline")
    
    # Create progress container
    progress_container = st.container()
    status_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Set up OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Mock processing steps
        steps = [
            ("📥 Downloading video and audio...", 20),
            ("🎞️ Extracting keyframes...", 40),
            ("🎤 Transcribing audio...", 60),
            ("👁️ Analyzing visual content...", 80),
        ]
        
        for step_text, progress in steps:
            with status_container:
                with st.status(step_text, expanded=True) as status:
                    # Simulate processing time
                    time.sleep(0.5)
                    status.update(label=f"✅ {step_text.replace('...', ' completed!')}", state="complete")
            
            progress_bar.progress(progress)
            time.sleep(0.2)
        
        # Real GPT API call for summary generation
        with status_container:
            with st.status("🧠 Generating summary with GPT...", expanded=True) as status:
                try:
                    # Mock transcript and visual content
                    mock_transcript = """
                    Welcome to this comprehensive guide on machine learning and artificial intelligence. 
                    Today we'll be covering the fundamentals of supervised learning, unsupervised learning, 
                    and neural networks. The presenter demonstrates practical examples using Python code 
                    and visual diagrams to illustrate complex concepts. We'll explore how machine learning 
                    algorithms work, from simple linear regression to complex deep learning models. 
                    The tutorial includes interactive elements and real-world applications of machine 
                    learning algorithms in various industries.
                    """
                    
                    mock_visual_content = """
                    [00:00:00] a person speaking at a whiteboard (confidence: 0.85)
                    [00:00:05] a presentation with mathematical formulas (confidence: 0.92)
                    [00:00:10] a computer screen showing code (confidence: 0.78)
                    [00:00:15] a demonstration of neural network visualization (confidence: 0.89)
                    [00:00:20] a graph showing training progress (confidence: 0.91)
                    """
                    
                    # Construct prompt
                    format_instructions = (
                        "Create a timestamped summary with key points and their timestamps."
                        if output_format == "timestamped"
                        else "Create a comprehensive summary of the video content."
                    )
                    
                    prompt = f"""Please analyze the following video content and create a summary.

Transcript:
{mock_transcript}

Visual Content Analysis:
{mock_visual_content}

Instructions:
{format_instructions}
Focus on the main points and key insights.
If there are any demonstrations or visual elements, make sure to mention them.
Keep the summary clear and concise while maintaining all important information.

Summary:"""
                    
                    # Make API call
                    response = openai.ChatCompletion.create(
                        model=model_choice,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that creates comprehensive video summaries."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=0.7
                    )
                    
                    summary = response.choices[0].message.content.strip()
                    
                    # Show API response details
                    st.info(f"✅ API Call Successful!")
                    st.info(f"Model: {model_choice}")
                    st.info(f"Tokens used: {response.usage.total_tokens}")
                    st.info(f"Prompt tokens: {response.usage.prompt_tokens}")
                    st.info(f"Completion tokens: {response.usage.completion_tokens}")
                    
                    status.update(label="✅ Summary generated with GPT!", state="complete")
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")
                    summary = f"Error generating summary: {str(e)}"
                    status.update(label="❌ Summary generation failed", state="error")
        
        progress_bar.progress(100)
        
        # Display results
        with results_container:
            st.markdown("---")
            st.header("📋 Generated Summary")
            
            # Summary display
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(summary)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                # Download summary as text
                st.download_button(
                    label="📄 Download Summary",
                    data=summary,
                    file_name=f"youtube_summary_{int(time.time())}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Show processing details
                with st.expander("📊 Processing Details"):
                    st.write(f"**Video URL:** {url}")
                    st.write(f"**Model used:** {model_choice}")
                    st.write(f"**Summary format:** {output_format}")
                    st.write(f"**Max tokens:** {max_tokens}")
                    st.write(f"**Summary length:** {len(summary.split())} words")
            
            # Show transcript preview
            with st.expander("📝 Transcript Preview"):
                st.text_area("Transcript", mock_transcript, height=200, disabled=True)
            
            # Show visual analysis preview
            with st.expander("👁️ Visual Analysis Preview"):
                st.text_area("Visual Content", mock_visual_content, height=200, disabled=True)
                
        except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.info("Check your API key and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit, OpenAI, and CLIP</p>
    <p>Transform any YouTube video into a comprehensive summary</p>
    <p><em>Testing GPT API integration - other components are mocked</em></p>
</div>
""", unsafe_allow_html=True) 