#!/usr/bin/env python3
"""
Vision Trial Script - Tests vision analysis capabilities.
Compares API-based (OpenAI Vision) vs local models for CPU-only systems.
"""

import os
import time
import base64
from PIL import Image
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

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

def create_mock_image():
    """Use the local living room image for testing."""
    try:
        # Path to the local living room image
        image_path = os.path.join(os.path.dirname(__file__), "living_room.jpg")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Load the image to get its properties
        img = Image.open(image_path)
        
        print(f"✅ Using local living room image: {image_path}")
        print(f"   Size: {img.size[0]}x{img.size[1]} pixels")
        print(f"   Format: {img.format}")
        print(f"   Mode: {img.mode}")
        
        return image_path
        
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return None

def encode_image_to_base64(image_path):
    """Encode image to base64 for API calls."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error encoding image: {str(e)}")
        return None

def test_openai_vision_api(client, image_path):
    """Test OpenAI Vision API for image analysis."""
    try:
        print("🧪 Testing OpenAI Vision API...")
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return False
        
        # Test different types of analysis
        analyses = [
            {
                "name": "General Description",
                "prompt": "Describe what you see in this living room image in detail. What furniture, decor, and overall style do you observe?"
            },
            {
                "name": "Scene Classification", 
                "prompt": "What type of room or setting is this? Is it a living room, family room, lounge, or something else? What makes you think that?"
            },
            {
                "name": "Content Analysis",
                "prompt": "What visual elements and objects can you identify? Look for furniture, decorations, colors, lighting, and any other notable features."
            },
            {
                "name": "Style Analysis",
                "prompt": "What design style or aesthetic does this room represent? Is it modern, traditional, minimalist, cozy, or something else? What elements suggest this style?"
            }
        ]
        
        results = {}
        
        for analysis in analyses:
            print(f"  📸 Testing: {analysis['name']}")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis["prompt"]
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            results[analysis["name"]] = {
                "result": result,
                "tokens": response.usage.total_tokens,
                "cost_estimate": response.usage.total_tokens * 0.00001  # Rough estimate
            }
            
            print(f"    ✅ {analysis['name']}: {result[:100]}...")
            print(f"    📊 Tokens: {response.usage.total_tokens}")
        
        # Display comprehensive results
        print("\n" + "="*60)
        print("OPENAI VISION API RESULTS:")
        print("="*60)
        
        total_tokens = 0
        total_cost = 0
        
        for name, data in results.items():
            print(f"\n🔍 {name}:")
            print(f"   Result: {data['result']}")
            print(f"   Tokens: {data['tokens']}")
            print(f"   Cost: ~${data['cost_estimate']:.4f}")
            total_tokens += data['tokens']
            total_cost += data['cost_estimate']
        
        print(f"\n📊 TOTAL:")
        print(f"   Total Tokens: {total_tokens}")
        print(f"   Estimated Cost: ~${total_cost:.4f}")
        print(f"   Response Time: Fast (API-based)")
        print(f"   Hardware: No local GPU required")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing OpenAI Vision API: {str(e)}")
        return False

def test_local_vision_analysis():
    """Test local vision analysis options for CPU-only systems."""
    print("\n🧪 Testing Local Vision Analysis Options...")
    
    local_options = [
        {
            "name": "Pillow (Basic)",
            "description": "Basic image processing and analysis",
            "pros": ["No API costs", "Fast", "No internet required"],
            "cons": ["Limited analysis", "No semantic understanding"],
            "setup": "pip install Pillow",
            "complexity": "Low"
        },
        {
            "name": "OpenCV (Intermediate)",
            "description": "Computer vision library with pre-trained models",
            "pros": ["Good for object detection", "No API costs", "Offline"],
            "cons": ["Requires model downloads", "Limited semantic analysis"],
            "setup": "pip install opencv-python",
            "complexity": "Medium"
        },
        {
            "name": "Transformers (CPU)",
            "description": "Run vision models locally on CPU",
            "pros": ["Advanced analysis", "No API costs", "Privacy"],
            "cons": ["Slow on CPU", "Large model sizes", "Memory intensive"],
            "setup": "pip install transformers torch",
            "complexity": "High"
        }
    ]
    
    print("\n" + "="*60)
    print("LOCAL VISION ANALYSIS OPTIONS:")
    print("="*60)
    
    for option in local_options:
        print(f"\n🔧 {option['name']}:")
        print(f"   Description: {option['description']}")
        print(f"   Setup: {option['setup']}")
        print(f"   Complexity: {option['complexity']}")
        print(f"   Pros: {', '.join(option['pros'])}")
        print(f"   Cons: {', '.join(option['cons'])}")
    
    return local_options

def compare_approaches():
    """Compare API vs Local approaches."""
    print("\n" + "="*60)
    print("APPROACH COMPARISON:")
    print("="*60)
    
    comparison = {
        "OpenAI Vision API": {
            "Speed": "Fast",
            "Cost": "~$0.01-0.05 per image",
            "Quality": "Excellent",
            "Setup": "API key only",
            "Hardware": "No GPU required",
            "Privacy": "Data sent to OpenAI",
            "Best For": "Production, high-quality analysis"
        },
        "Local CPU Models": {
            "Speed": "Slow (30s-2min per image)",
            "Cost": "Free",
            "Quality": "Good (depends on model)",
            "Setup": "Complex (model downloads)",
            "Hardware": "CPU only (slow)",
            "Privacy": "100% local",
            "Best For": "Privacy-sensitive, offline use"
        }
    }
    
    for approach, details in comparison.items():
        print(f"\n🎯 {approach}:")
        for metric, value in details.items():
            print(f"   {metric}: {value}")

def test_simple_local_analysis(image_path):
    """Test basic local image analysis using Pillow."""
    try:
        print("\n🧪 Testing Basic Local Analysis (Pillow)...")
        
        from PIL import Image
        
        # Load image
        img = Image.open(image_path)
        
        # Basic analysis
        analysis = {
            "Size": f"{img.size[0]}x{img.size[1]} pixels",
            "Mode": img.mode,
            "Format": img.format,
            "File Size": f"{os.path.getsize(image_path)} bytes"
        }
        
        print("✅ Basic Local Analysis Results:")
        for key, value in analysis.items():
            print(f"   {key}: {value}")
        
        print(f"   Speed: Very Fast (<1s)")
        print(f"   Cost: Free")
        print(f"   Hardware: CPU only")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic local analysis: {str(e)}")
        return False

def main():
    """Main function to run all vision tests."""
    print("🚀 Starting Vision Analysis Trial Tests")
    print("="*60)
    
    # Set up OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Test 1: API Connection
    if not test_openai_connection():
        return
    
    # Test 2: Create mock image
    image_path = create_mock_image()
    if not image_path:
        return
    
    try:
        # Test 3: OpenAI Vision API
        test_openai_vision_api(client, image_path)
        
        # Test 4: Local analysis options
        test_local_vision_analysis()
        
        # Test 5: Simple local analysis
        test_simple_local_analysis(image_path)
        
        # Test 6: Comparison
        compare_approaches()
        
        print("\n" + "="*60)
        print("🎉 Vision Analysis Trial Tests Completed!")
        print("="*60)
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("   • For production: Use OpenAI Vision API")
        print("   • For privacy/offline: Use local models (but expect slower performance)")
        print("   • For basic analysis: Use Pillow + OpenCV")
        print("   • For your use case: Start with API, add local fallback later")
        
    finally:
        # No cleanup needed - using local image
        print(f"\n📁 Image preserved: {image_path}")

if __name__ == "__main__":
    main()
