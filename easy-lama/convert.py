
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
from .model import LamaModel

def convert_pytorch_to_safetensors(input_path: str, output_path: str = None):
    """Convert PyTorch checkpoint to safetensors format."""
    
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.safetensors')
    else:
        output_path = Path(output_path)
    
    print(f"Converting {input_path} to {output_path}")
    
    # Load PyTorch checkpoint
    print("Loading PyTorch checkpoint...")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Clean up state dict keys if needed
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes
        clean_key = key
        if key.startswith('module.'):
            clean_key = key[7:]  # Remove 'module.' prefix
        if key.startswith('model.'):
            clean_key = key[6:]   # Remove 'model.' prefix
        
        cleaned_state_dict[clean_key] = value
    
    # Validate by loading into model
    print("Validating state dict...")
    try:
        model = LamaModel()
        model.load_state_dict(cleaned_state_dict)
        print("✓ State dict is valid")
    except Exception as e:
        print(f"✗ Warning: State dict validation failed: {e}")
        print("Proceeding anyway...")
    
    # Save as safetensors
    print("Saving as safetensors...")
    save_file(cleaned_state_dict, output_path)
    
    # Verify the saved file
    print("Verifying saved file...")
    try:
        from safetensors.torch import load_file
        loaded_dict = load_file(output_path)
        print(f"✓ Successfully saved {len(loaded_dict)} parameters")
        
        # Compare sizes
        original_size = input_path.stat().st_size / (1024 * 1024)  # MB
        new_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"Original size: {original_size:.1f} MB")
        print(f"Safetensors size: {new_size:.1f} MB")
        print(f"Size change: {((new_size - original_size) / original_size * 100):+.1f}%")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False
    
    print(f"✓ Conversion completed successfully!")
    return True

def main():
    """CLI for converting PyTorch models to safetensors."""
    parser = argparse.ArgumentParser(description="Convert LAMA PyTorch models to safetensors format")
    parser.add_argument("input", help="Input PyTorch model file (.pt or .pth)")
    parser.add_argument("-o", "--output", help="Output safetensors file (optional)")
    parser.add_argument("--verify", action="store_true", help="Verify conversion by loading both files")
    
    args = parser.parse_args()
    
    try:
        success = convert_pytorch_to_safetensors(args.input, args.output)
        
        if args.verify and success:
            print("\nPerforming additional verification...")
            # Load both files and compare
            original = torch.load(args.input, map_location='cpu')
            if 'state_dict' in original:
                original = original['state_dict']
            
            from safetensors.torch import load_file
            converted = load_file(args.output or Path(args.input).with_suffix('.safetensors'))
            
            # Compare keys
            orig_keys = set(original.keys())
            conv_keys = set(converted.keys())
            
            if orig_keys == conv_keys:
                print("✓ All keys match")
            else:
                print("✗ Key mismatch!")
                print(f"Missing: {orig_keys - conv_keys}")
                print(f"Extra: {conv_keys - orig_keys}")
            
            # Compare a few tensors
            matching_keys = orig_keys & conv_keys
            for key in list(matching_keys)[:3]:  # Check first 3 tensors
                if torch.allclose(original[key], converted[key]):
                    print(f"✓ {key}: tensors match")
                else:
                    print(f"✗ {key}: tensors differ!")
        
        exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()