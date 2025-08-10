#!/usr/bin/env python3
"""
Trial script to test LLM functionality only.
This script tests the OpenAI API integration without requiring video or vision models.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def test_openai_connection():
    """Test basic OpenAI API connection."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file")
        return False
    
    print("✅ OpenAI API key found")
    return True

def test_simple_completion(client, model="gpt-4o-mini"):
    """Test a simple completion to verify API works."""
    try:
        print(f"🧪 Testing {model} with simple completion...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'Hello, LLM is working!' in a creative way."}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ {model} Response: {result}")
        print(f"📊 Tokens used: {response.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model}: {str(e)}")
        return False

def test_summary_generation(client, model="gpt-4o-mini"):
    """Test summary generation with mock data."""
    try:
        print(f"🧪 Testing {model} with mock summary generation...")
        
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
        
        prompt = f"""Please analyze the following video content and create a comprehensive summary.

Transcript:
{mock_transcript}

Visual Content Analysis:
{mock_visual_content}

Instructions:
Create a comprehensive summary of the video content.
Focus on the main points and key insights.
If there are any demonstrations or visual elements, make sure to mention them.
Keep the summary clear and concise while maintaining all important information.

Summary:"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates comprehensive video summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        
        print(f"✅ {model} Summary Generated Successfully!")
        print(f"📊 Tokens used: {response.usage.total_tokens}")
        print(f"📝 Prompt tokens: {response.usage.prompt_tokens}")
        print(f"📝 Completion tokens: {response.usage.completion_tokens}")
        print("\n" + "="*50)
        print("GENERATED SUMMARY:")
        print("="*50)
        print(summary)
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model} summary generation: {str(e)}")
        return False

def test_timestamped_summary(client, model="gpt-4o-mini"):
    """Test timestamped summary generation."""
    try:
        print(f"🧪 Testing {model} with timestamped summary...")
        
        mock_transcript = """
        [00:00:00] Welcome to this machine learning tutorial
        [00:00:15] Today we'll cover supervised learning
        [00:00:30] Let me show you some practical examples
        [00:00:45] Here's how we implement neural networks
        [00:01:00] This is the training process demonstration
        """
        
        mock_visual_content = """
        [00:00:00] presenter at whiteboard (confidence: 0.85)
        [00:00:15] mathematical formulas on screen (confidence: 0.92)
        [00:00:30] code demonstration (confidence: 0.78)
        [00:00:45] neural network diagram (confidence: 0.89)
        [00:01:00] training progress graph (confidence: 0.91)
        """
        
        prompt = f"""Please analyze the following video content and create a timestamped summary.

Transcript:
{mock_transcript}

Visual Content Analysis:
{mock_visual_content}

Instructions:
Create a timestamped summary with key points and their timestamps.
Focus on the main points and key insights.
If there are any demonstrations or visual elements, make sure to mention them.
Keep the summary clear and concise while maintaining all important information.

Summary:"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates comprehensive video summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        
        print(f"✅ {model} Timestamped Summary Generated Successfully!")
        print(f"📊 Tokens used: {response.usage.total_tokens}")
        print("\n" + "="*50)
        print("TIMESTAMPED SUMMARY:")
        print("="*50)
        print(summary)
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model} timestamped summary: {str(e)}")
        return False



def main():
    """Main function to run all tests."""
    print("🚀 Starting LLM Trial Tests")
    print("="*50)
    
    # Set up OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Test 1: API Connection
    if not test_openai_connection():
        return
    
    print("\n" + "="*50)
    
    # Test 2: Simple completion
    test_simple_completion(client)
    
    print("\n" + "="*50)
    
    # Test 3: Summary generation
    test_summary_generation(client)
    
    print("\n" + "="*50)
    
    # Test 4: Timestamped summary
    test_timestamped_summary(client)
    
    print("\n" + "="*50)
    print("🎉 LLM Trial Tests Completed!")
    print("="*50)

if __name__ == "__main__":
    main()
