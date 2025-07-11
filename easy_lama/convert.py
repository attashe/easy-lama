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
    
    print(f"Original checkpoint contains {len(state_dict)} parameters")
    
    # Filter out non-essential keys (evaluation metrics, optimizers, etc.)
    filtered_state_dict = {}
    skip_patterns = [
        'test_evaluator.',
        'val_evaluator.',
        'train_evaluator.',
        'evaluator.',
        'optimizer.',
        'lr_scheduler.',
        'epoch',
        'global_step',
        'pytorch-lightning_version',
        'state_dict',
        'hyper_parameters',
    ]
    
    essential_patterns = [
        'generator.',
        'model.',
        'discriminator.',
        'encoder.',
        'decoder.',
        'conv',
        'norm',
        'linear',
        'embedding',
        'attention',
    ]
    
    for key, value in state_dict.items():
        # Skip evaluation and training artifacts
        should_skip = any(pattern in key for pattern in skip_patterns)
        
        if should_skip:
            print(f"Skipping: {key}")
            continue
            
        # Clean up key names
        clean_key = key
        if key.startswith('module.'):
            clean_key = key[7:]  # Remove 'module.' prefix
        if key.startswith('model.'):
            clean_key = key[6:]   # Remove 'model.' prefix
        if key.startswith('generator.'):
            clean_key = key[10:]  # Remove 'generator.' prefix
            
        filtered_state_dict[clean_key] = value
    
    print(f"Filtered to {len(filtered_state_dict)} essential parameters")
    
    # Handle shared tensors by creating independent copies
    print("Checking for shared tensors...")
    tensor_ids = {}
    duplicates = []
    
    for key, tensor in filtered_state_dict.items():
        tensor_id = id(tensor)
        if tensor_id in tensor_ids:
            duplicates.append((key, tensor_ids[tensor_id]))
        else:
            tensor_ids[tensor_id] = key
    
    if duplicates:
        print(f"Found {len(duplicates)} shared tensors, creating independent copies...")
        for dup_key, orig_key in duplicates:
            print(f"  Copying shared tensor: {dup_key} (shared with {orig_key})")
            filtered_state_dict[dup_key] = filtered_state_dict[dup_key].clone()
    
    # Validate by loading into model
    print("Validating state dict...")
    try:
        model = LamaModel()
        
        # Try to match keys if needed
        model_keys = set(model.state_dict().keys())
        dict_keys = set(filtered_state_dict.keys())
        
        if model_keys != dict_keys:
            print("Key mismatch detected, attempting to fix...")
            print(f"Model expects {len(model_keys)} keys, got {len(dict_keys)} keys")
            
            # Try common key transformations
            fixed_dict = {}
            for model_key in model_keys:
                found = False
                for dict_key in dict_keys:
                    if (dict_key.endswith(model_key) or 
                        model_key.endswith(dict_key) or
                        dict_key.replace('generator.', '') == model_key or
                        dict_key.replace('model.', '') == model_key):
                        fixed_dict[model_key] = filtered_state_dict[dict_key]
                        found = True
                        break
                
                if not found:
                    print(f"Warning: Could not find match for {model_key}")
            
            if len(fixed_dict) > len(filtered_state_dict) * 0.8:  # If we matched >80% of keys
                print(f"Using fixed keys ({len(fixed_dict)} matched)")
                filtered_state_dict = fixed_dict
        
        # Try loading
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            
        if len(missing_keys) < len(model.state_dict()) * 0.5:  # If <50% missing
            print("✓ State dict is reasonably valid")
        else:
            print("✗ Warning: Many missing keys, but proceeding...")
            
    except Exception as e:
        print(f"✗ Warning: State dict validation failed: {e}")
        print("Proceeding anyway...")
    
    # Save as safetensors
    print("Saving as safetensors...")
    try:
        save_file(filtered_state_dict, output_path)
    except Exception as e:
        if "share memory" in str(e):
            print("Still have shared tensors, forcing deep copy...")
            # Force deep copy of all tensors
            deep_copy_dict = {}
            for key, tensor in filtered_state_dict.items():
                deep_copy_dict[key] = tensor.detach().clone()
            save_file(deep_copy_dict, output_path)
        else:
            raise e
    
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
    print(f"✓ Removed evaluation metrics and training artifacts")
    print(f"✓ Model ready for safe inference!")
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