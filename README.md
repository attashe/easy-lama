# LAMA Inpainting - Clean Implementation

A streamlined PyPI package for LAMA (Large Mask Inpainting) inference with minimal dependencies and clean API.

## Features

- **Simple API**: Just 2 lines of code for inpainting
- **🔒 Safetensors Support**: Safe model format by default (no arbitrary code execution)
- **Automatic model downloading**: Models are downloaded on first use
- **Flexible input formats**: Supports numpy arrays and PIL Images
- **GPU acceleration**: Automatic GPU detection with CPU fallback
- **High resolution support**: Works well on images up to 2K+ resolution
- **Model conversion**: Convert PyTorch models to safetensors format

## Installation

```bash
pip install lama-inpainting
```

## Quick Start

```python
from lama_inpainting import TextureInpainter
from PIL import Image
import numpy as np

# Initialize inpainter
inpainter = TextureInpainter()

# Load your image and mask
image = Image.open("your_image.jpg")
mask = Image.open("your_mask.png")  # White areas will be inpainted

# Inpaint
result = inpainter.inpaint(image, mask)

# Save result
Image.fromarray(result).save("inpainted.jpg")
```

## API Reference

### TextureInpainter

```python
inpainter = TextureInpainter(device="auto", model_size="big", use_safetensors=True)
```

**Parameters:**
- `device`: Device to run on ("auto", "cpu", "cuda")
- `model_size`: Model size ("big" for better quality)
- `use_safetensors`: Use safetensors format for safety (recommended: True)

### inpaint

```python
result = inpainter.inpaint(image, mask)
```

**Parameters:**
- `image`: Input image as numpy array (H,W,3) or PIL Image
- `mask`: Binary mask as numpy array (H,W) or PIL Image. White pixels (255) will be inpainted

**Returns:**
- Inpainted image as numpy array (H,W,3) with values 0-255

## Examples

### Remove objects from photos

```python
import numpy as np
from PIL import Image
from lama_inpainting import TextureInpainter

# Create inpainter (uses safetensors by default for safety)
inpainter = TextureInpainter()

# Load image
image = np.array(Image.open("photo.jpg"))

# Create mask (you can create this manually or use segmentation tools)
mask = np.array(Image.open("object_mask.png").convert('L'))

# Remove object
result = inpainter.inpaint(image, mask)

# Save
Image.fromarray(result).save("photo_cleaned.jpg")
```

### Force PyTorch format (if needed)

```python
from lama_inpainting import TextureInpainter

# Use PyTorch format instead of safetensors
inpainter = TextureInpainter(use_safetensors=False)
result = inpainter.inpaint(image, mask)
```

### Convert existing models to safetensors

```python
from lama_inpainting.convert import convert_pytorch_to_safetensors

# Convert a PyTorch model to safetensors
success = convert_pytorch_to_safetensors("my_model.pt", "my_model.safetensors")
if success:
    print("Conversion successful!")
```

### Restore damaged photos

```python
from lama_inpainting import TextureInpainter

inpainter = TextureInpainter()

# Load damaged photo and damage mask
photo = Image.open("old_photo.jpg")
damage_mask = Image.open("damage_areas.png")

# Restore
restored = inpainter.inpaint(photo, damage_mask)

# Save restored photo
Image.fromarray(restored).save("restored_photo.jpg")
```

## Command Line Interface

### Inpainting

```bash
# Basic usage (uses safetensors by default for safety)
lama-inpaint input.jpg mask.png output.jpg

# Specify device and model
lama-inpaint input.jpg mask.png output.jpg --device cuda --model-size big

# Force PyTorch format instead of safetensors
lama-inpaint input.jpg mask.png output.jpg --no-safetensors
```

### Model Conversion

Convert PyTorch models to safer safetensors format:

```bash
# Convert PyTorch model to safetensors
lama-convert model.pt

# Specify output file
lama-convert model.pt -o model_safe.safetensors

# Verify conversion
lama-convert model.pt --verify
```

## 🔒 Safetensors vs PyTorch Format

**Safetensors** (recommended):
- ✅ **Safe**: No arbitrary code execution
- ✅ **Fast**: Faster loading than PyTorch
- ✅ **Smaller**: Usually smaller file size
- ✅ **Cross-platform**: Better compatibility

**PyTorch format**:
- ⚠️ **Risk**: Can execute arbitrary code when loaded
- ⚠️ **Slower**: Pickle-based loading
- ⚠️ **Larger**: Usually larger file size

By default, the library uses safetensors format for your safety.

## License

Apache License 2.0

## Citation

This implementation is based on the paper:

```
@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}
```
