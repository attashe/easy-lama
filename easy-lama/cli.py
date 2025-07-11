import argparse
from pathlib import Path
from PIL import Image
from .inpainter import TextureInpainter

def main():
    """Command-line interface for LAMA inpainting."""
    parser = argparse.ArgumentParser(description="LAMA inpainting CLI")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("mask", help="Input mask path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Device to use")
    parser.add_argument("--model-size", default="big", choices=["big", "regular"],
                       help="Model size")
    parser.add_argument("--use-safetensors", action="store_true", default=True,
                       help="Use safetensors format (safer)")
    parser.add_argument("--no-safetensors", action="store_true",
                       help="Force PyTorch format instead of safetensors")
    
    args = parser.parse_args()
    
    # Handle safetensors preference
    use_safetensors = args.use_safetensors and not args.no_safetensors
    
    # Load images
    print(f"Loading image: {args.image}")
    image = Image.open(args.image)
    
    print(f"Loading mask: {args.mask}")
    mask = Image.open(args.mask).convert('L')
    
    # Initialize inpainter
    format_str = "safetensors" if use_safetensors else "PyTorch"
    print(f"Initializing inpainter (device: {args.device}, model: {args.model_size}, format: {format_str})")
    inpainter = TextureInpainter(device=args.device, model_size=args.model_size, use_safetensors=use_safetensors)
    
    # Inpaint
    print("Running inpainting...")
    result = inpainter.inpaint(image, mask)
    
    # Save result
    print(f"Saving result: {args.output}")
    Image.fromarray(result).save(args.output)
    print("Done!")

if __name__ == "__main__":
    main()