
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
import os
from pathlib import Path
import zipfile
from typing import Union, Tuple

def download_model(model_size: str = "big", use_safetensors: bool = True) -> str:
    """Download LAMA model if not exists, with safetensors support."""
    model_dir = Path.home() / ".cache" / "lama_inpainting"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if use_safetensors:
        # Safetensors URLs (safer format)
        if model_size == "big":
            model_url = "https://huggingface.co/smartywu/lama-big-safetensors/resolve/main/model.safetensors"
            model_file = model_dir / "big-lama.safetensors"
        else:
            model_url = "https://huggingface.co/smartywu/lama-regular-safetensors/resolve/main/model.safetensors"
            model_file = model_dir / "lama.safetensors"
    else:
        # Original PyTorch format
        if model_size == "big":
            model_url = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
            model_file = model_dir / "big-lama.pt"
        else:
            model_url = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
            model_file = model_dir / "lama.pt"
    
    if not model_file.exists():
        format_name = "safetensors" if use_safetensors else "PyTorch"
        print(f"Downloading LAMA model ({model_size}, {format_name} format)...")
        
        try:
            if model_url.endswith('.zip'):
                # Download and extract
                zip_path = model_dir / "model.zip"
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
                
                # Find the actual model file
                for file in model_dir.rglob("*.pt"):
                    file.rename(model_file)
                    break
                
                zip_path.unlink()  # Remove zip file
            else:
                # Direct download
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                with open(model_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print(f"Model downloaded successfully! ({format_name} format)")
            
        except Exception as e:
            if use_safetensors:
                print(f"Failed to download safetensors model: {e}")
                print("Falling back to PyTorch format...")
                return download_model(model_size, use_safetensors=False)
            else:
                raise e
    
    return str(model_file)

def preprocess_image(image: Union[np.ndarray, Image.Image], 
                    mask: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Preprocess image and mask for inference."""
    
    # Convert to PIL Images
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    
    # Ensure RGB and L modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # Store original size
    orig_size = image.size  # (W, H)
    
    # Convert to numpy arrays
    img_np = np.array(image).astype(np.float32) / 255.0
    mask_np = np.array(mask).astype(np.float32) / 255.0
    
    # Convert to tensors (C, H, W)
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
    
    return img_tensor, mask_tensor, (orig_size[1], orig_size[0])  # (H, W)

def pad_to_modulo(img_tensor: torch.Tensor, mask_tensor: torch.Tensor, 
                  modulo: int = 8) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Pad tensors to be divisible by modulo."""
    _, h, w = img_tensor.shape
    
    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo
    
    if pad_h > 0 or pad_w > 0:
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        img_tensor = F.pad(img_tensor, padding, mode='reflect')
        mask_tensor = F.pad(mask_tensor, padding, mode='reflect')
        
        pad_info = {'pad_h': pad_h, 'pad_w': pad_w}
    else:
        pad_info = {'pad_h': 0, 'pad_w': 0}
    
    return img_tensor, mask_tensor, pad_info

def postprocess_result(output: torch.Tensor, orig_size: Tuple[int, int], 
                      pad_info: dict) -> np.ndarray:
    """Postprocess model output to final result."""
    
    # Remove padding
    if pad_info['pad_h'] > 0 or pad_info['pad_w'] > 0:
        h_end = output.shape[1] - pad_info['pad_h'] if pad_info['pad_h'] > 0 else output.shape[1]
        w_end = output.shape[2] - pad_info['pad_w'] if pad_info['pad_w'] > 0 else output.shape[2]
        output = output[:, :h_end, :w_end]
    
    # Convert to numpy
    result = output.detach().cpu().numpy()
    result = np.transpose(result, (1, 2, 0))  # (H, W, C)
    
    # Resize to original size if needed
    if result.shape[:2] != orig_size:
        result = np.array(Image.fromarray((result * 255).astype(np.uint8)).resize((orig_size[1], orig_size[0])))
        result = result.astype(np.float32) / 255.0
    
    # Clip and convert to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result