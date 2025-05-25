import openai
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SummaryGenerator:
    def __init__(self):
        """Initialize the summary generator with OpenAI API key."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        openai.api_key = self.api_key

    def generate_summary(
        self,
        transcript: str,
        visual_content: str,
        format_type: str = "long",
        max_tokens: int = 1000
    ) -> str:
        """Generate a comprehensive summary combining transcript and visual content.
        
        Args:
            transcript (str): Formatted transcript
            visual_content (str): Formatted visual content analysis
            format_type (str): Output format type ("long" or "timestamped")
            max_tokens (int): Maximum tokens for the summary
            
        Returns:
            str: Generated summary
        """
        # Construct the prompt
        prompt = self._construct_prompt(transcript, visual_content, format_type)
        
        try:
            # Generate summary using GPT-4
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates comprehensive video summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")

    def _construct_prompt(
        self,
        transcript: str,
        visual_content: str,
        format_type: str
    ) -> str:
        """Construct the prompt for the LLM.
        
        Args:
            transcript (str): Formatted transcript
            visual_content (str): Formatted visual content analysis
            format_type (str): Output format type
            
        Returns:
            str: Constructed prompt
        """
        format_instructions = (
            "Create a timestamped summary with key points and their timestamps."
            if format_type == "timestamped"
            else "Create a comprehensive summary of the video content."
        )
        
        return f"""Please analyze the following video content and create a summary.

Transcript:
{transcript}

Visual Content Analysis:
{visual_content}

Instructions:
{format_instructions}
Focus on the main points and key insights.
If there are any demonstrations or visual elements, make sure to mention them.
Keep the summary clear and concise while maintaining all important information.

Summary:""" 