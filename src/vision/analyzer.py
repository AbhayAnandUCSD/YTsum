import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Tuple, Dict
import numpy as np

class VisionAnalyzer:
    def __init__(self):
        """Initialize the vision analyzer with CLIP model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Common scene descriptions for better context
        self.scene_descriptions = [
            "a person speaking", "a presentation", "a demonstration",
            "a tutorial", "a lecture", "a product showcase",
            "a landscape", "a close-up shot", "a group of people",
            "a computer screen", "a whiteboard", "a classroom"
        ]

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame using CLIP.
        
        Args:
            frame (np.ndarray): RGB image array
            
        Returns:
            Dict: Analysis results with confidence scores
        """
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Process image and text inputs
        inputs = self.processor(
            images=image,
            text=self.scene_descriptions,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get image and text features
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top 3 most likely descriptions
        top_probs, top_indices = torch.topk(probs[0], 3)
        
        results = {
            "descriptions": [
                {
                    "text": self.scene_descriptions[idx],
                    "confidence": float(prob)
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
        }
        
        return results

    def analyze_keyframes(self, keyframes: List[Tuple[float, np.ndarray]]) -> List[Dict]:
        """Analyze multiple keyframes.
        
        Args:
            keyframes (List[Tuple[float, np.ndarray]]): List of (timestamp, frame) tuples
            
        Returns:
            List[Dict]: List of analysis results for each keyframe
        """
        results = []
        for timestamp, frame in keyframes:
            analysis = self.analyze_frame(frame)
            results.append({
                "timestamp": timestamp,
                "analysis": analysis
            })
        return results

    def format_visual_content(self, analyses: List[Dict]) -> str:
        """Format visual content analysis into a readable string.
        
        Args:
            analyses (List[Dict]): List of frame analyses
            
        Returns:
            str: Formatted visual content description
        """
        formatted = []
        for analysis in analyses:
            timestamp = self._format_timestamp(analysis["timestamp"])
            top_description = analysis["analysis"]["descriptions"][0]
            formatted.append(
                f"[{timestamp}] {top_description['text']} "
                f"(confidence: {top_description['confidence']:.2f})"
            )
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