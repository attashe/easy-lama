from easy_lama import TextureInpainter
from PIL import Image
import numpy as np

def main():
    """Example usage of the LAMA inpainting library with safetensors."""
    
    # Initialize the inpainter (downloads model on first use, uses safetensors by default)
    print("Initializing LAMA inpainter with safetensors (safe format)...")
    inpainter = TextureInpainter(use_safetensors=True)
    
    # Example 1: Using PIL Images
    print("Example 1: PIL Image input")
    image = Image.open("your_image.jpg")
    mask = Image.open("your_mask.png").convert('L')
    
    result = inpainter.inpaint(image, mask)
    Image.fromarray(result).save("result_pil.jpg")
    print("Saved result_pil.jpg")
    
    # Example 2: Using numpy arrays
    print("Example 2: Numpy array input")
    texture_np = np.array(image)  # (H, W, 3)
    mask_np = np.array(mask)      # (H, W)
    
    # This is exactly what the user requested!
    texture_np = inpainter.inpaint(texture_np, mask_np)
    
    Image.fromarray(texture_np).save("result_numpy.jpg")
    print("Saved result_numpy.jpg")
    
    # Example 3: Simple test with generated mask
    print("Example 3: Simple test with generated mask")
    # Create a test image
    test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Create a mask (white square in center will be inpainted)
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[200:312, 200:312] = 255  # White square
    
    result = inpainter.inpaint(test_img, test_mask)
    Image.fromarray(result).save("test_result.jpg")
    print("Saved test_result.jpg")
    
    # Example 4: Using PyTorch format (if needed)
    print("Example 4: Using PyTorch format instead of safetensors")
    try:
        pytorch_inpainter = TextureInpainter(use_safetensors=False)
        result_pt = pytorch_inpainter.inpaint(test_img, test_mask)
        Image.fromarray(result_pt).save("test_pytorch.jpg")
        print("Saved test_pytorch.jpg")
    except Exception as e:
        print(f"PyTorch format failed: {e}")
    
    # Example 5: Convert existing model (if you have one)
    print("Example 5: Model conversion (if you have a PyTorch model)")
    try:
        from lama_inpainting.convert import convert_pytorch_to_safetensors
        # convert_pytorch_to_safetensors("my_model.pt", "my_model.safetensors")
        print("Use: lama-convert my_model.pt")
    except ImportError:
        print("Conversion module not available")

if __name__ == "__main__":
    main()
