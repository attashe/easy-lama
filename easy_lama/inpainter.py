import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import yaml
from pathlib import Path
from typing import Union, Tuple
import requests
import zipfile
from safetensors.torch import load_file as load_safetensors
from .model import LamaModel
from .utils import download_model, preprocess_image, postprocess_result, pad_to_modulo

class TextureInpainter:
    """Simple LAMA inpainting interface with safetensors support."""
    
    def __init__(self, device: str = "auto", model_size: str = "big", use_safetensors: bool = True):
        """
        Initialize the LAMA inpainter.
        
        Args:
            device: Device to run on ("auto", "cpu", "cuda")
            model_size: Model size ("big" or "regular")
            use_safetensors: Whether to use safetensors format for safety
        """
        self.device = self._get_device(device)
        self.model_size = model_size
        self.use_safetensors = use_safetensors
        self.model = None
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load the LAMA model with safetensors support."""
        model_path = download_model(self.model_size, use_safetensors=self.use_safetensors)
        self.model = LamaModel()
        
        # Load checkpoint based on format
        if model_path.endswith('.safetensors'):
            print("Loading safetensors model (safe format)...")
            state_dict = load_safetensors(model_path)
            self.model.load_state_dict(state_dict)
        else:
            print("Loading PyTorch model...")
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def inpaint(self, image: Union[np.ndarray, Image.Image], mask: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Inpaint image using mask.
        
        Args:
            image: Input image (H, W, 3) numpy array or PIL Image
            mask: Binary mask (H, W) numpy array or PIL Image, 255 for inpaint areas
            
        Returns:
            Inpainted image as numpy array (H, W, 3)
        """
        # Preprocess inputs
        img_tensor, mask_tensor, orig_size = preprocess_image(image, mask)
        
        # Pad to modulo for better performance
        img_tensor, mask_tensor, pad_info = pad_to_modulo(img_tensor, mask_tensor, 8)
        
        # Prepare batch
        batch = {
            'image': img_tensor.unsqueeze(0).to(self.device),
            'mask': mask_tensor.unsqueeze(0).to(self.device)
        }
        
        # Inference
        with torch.no_grad():
            batch['mask'] = (batch['mask'] > 0.5).float()
            result = self.model(batch)
            output = result['inpainted'][0]  # Remove batch dimension
        
        # Postprocess
        return postprocess_result(output, orig_size, pad_info)
