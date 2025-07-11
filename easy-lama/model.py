import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class FFCResnetBlock(nn.Module):
    """Fast Fourier Convolution Residual Block."""
    
    def __init__(self, dim: int, ratio_gin: float = 0.75, ratio_gout: float = 0.75):
        super().__init__()
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        
        # Calculate channel splits
        self.dim_in_l = int(dim * (1 - ratio_gin))
        self.dim_in_g = dim - self.dim_in_l
        self.dim_out_l = int(dim * (1 - ratio_gout))
        self.dim_out_g = dim - self.dim_out_l
        
        # Local convolution
        if self.dim_in_l > 0 and self.dim_out_l > 0:
            self.conv_l2l = nn.Sequential(
                nn.Conv2d(self.dim_in_l, self.dim_out_l, 3, 1, 1),
                nn.BatchNorm2d(self.dim_out_l),
                nn.ReLU(inplace=True)
            )
        
        # Global convolution (FFT-based)
        if self.dim_in_g > 0 and self.dim_out_g > 0:
            self.conv_g2g = nn.Sequential(
                nn.Conv2d(self.dim_in_g, self.dim_out_g, 1),
                nn.BatchNorm2d(self.dim_out_g),
                nn.ReLU(inplace=True)
            )
        
        # Cross connections
        if self.dim_in_l > 0 and self.dim_out_g > 0:
            self.conv_l2g = nn.Sequential(
                nn.Conv2d(self.dim_in_l, self.dim_out_g, 1),
                nn.BatchNorm2d(self.dim_out_g),
                nn.ReLU(inplace=True)
            )
        
        if self.dim_in_g > 0 and self.dim_out_l > 0:
            self.conv_g2l = nn.Sequential(
                nn.Conv2d(self.dim_in_g, self.dim_out_l, 1),
                nn.BatchNorm2d(self.dim_out_l),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l, x_g = self._split_channels(x)
        
        out_l, out_g = 0, 0
        
        if self.dim_in_l > 0 and self.dim_out_l > 0:
            out_l = out_l + self.conv_l2l(x_l)
        
        if self.dim_in_g > 0 and self.dim_out_g > 0:
            out_g = out_g + self._apply_fft_conv(x_g)
        
        if self.dim_in_l > 0 and self.dim_out_g > 0:
            out_g = out_g + self.conv_l2g(x_l)
        
        if self.dim_in_g > 0 and self.dim_out_l > 0:
            out_l = out_l + self.conv_g2l(x_g)
        
        return self._combine_channels(out_l, out_g) + x
    
    def _split_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input into local and global channels."""
        return x[:, :self.dim_in_l], x[:, self.dim_in_l:]
    
    def _combine_channels(self, x_l: torch.Tensor, x_g: torch.Tensor) -> torch.Tensor:
        """Combine local and global channels."""
        return torch.cat([x_l, x_g], dim=1)
    
    def _apply_fft_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFT-based convolution for global context."""
        # Simplified FFT convolution - in practice this would be more complex
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        x_fft = self.conv_g2g(torch.fft.irfft2(x_fft, dim=(-2, -1)))
        return x_fft

class LamaModel(nn.Module):
    """Streamlined LAMA model for inference."""
    
    def __init__(self, input_nc: int = 4, output_nc: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, 1, 0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            # Downsampling
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )
        
        # FFC Residual blocks
        self.middle = nn.Sequential(
            *[FFCResnetBlock(ngf * 4) for _ in range(n_blocks)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsampling
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            # Output
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        image = batch['image']
        mask = batch['mask']
        
        # Concatenate image and mask
        masked_img = image * (1 - mask)
        x = torch.cat([masked_img, mask], dim=1)
        
        # Encode
        x = self.encoder(x)
        
        # Process with FFC blocks
        x = self.middle(x)
        
        # Decode
        x = self.decoder(x)
        
        # Combine with original image
        result = image * (1 - mask) + x * mask
        
        return {'inpainted': result}