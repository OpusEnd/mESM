"""
Attention U-Net model for cell segmentation.

This module defines the Attention U-Net architecture for cell segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AttentionBlock(nn.Module):
    """
    Attention Gate module that helps the model focus on relevant features.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Initialize attention block parameters.
        
        Args:
            F_g: Number of feature channels from the decoder pathway
            F_l: Number of feature channels from the encoder pathway
            F_int: Number of intermediate channels
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        """
        Forward pass for attention block.
        
        Args:
            g: Feature map from decoder pathway
            x: Feature map from encoder pathway
            
        Returns:
            Attention-weighted feature map
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(torch.relu(g1 + x1))
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture for cell segmentation tasks.
    """
    def __init__(self, in_channels=3, out_channels=1):
        """
        Initialize Attention U-Net model.
        
        Args:
            in_channels: Number of input channels (default: 3)
            out_channels: Number of output channels (default: 1)
        """
        super(AttentionUNet, self).__init__()

        # Encoder (downsampling path)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attention4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attention3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attention2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attention1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Create a convolutional block with two conv layers.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            
        Returns:
            Sequential container with two convolutional layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass for Attention U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with segmentation predictions
        """
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        up4 = self.upconv4(b)
        att4 = self.attention4(up4, e4)
        d4 = self.decoder4(torch.cat([att4, up4], dim=1))

        up3 = self.upconv3(d4)
        att3 = self.attention3(up3, e3)
        d3 = self.decoder3(torch.cat([att3, up3], dim=1))

        up2 = self.upconv2(d3)
        att2 = self.attention2(up2, e2)
        d2 = self.decoder2(torch.cat([att2, up2], dim=1))

        up1 = self.upconv1(d2)
        att1 = self.attention1(up1, e1)
        d1 = self.decoder1(torch.cat([att1, up1], dim=1))

        # Final output
        out = self.final(d1)

        return out


def apply_crf(image, output):
    """
    Apply Conditional Random Field (CRF) as a post-processing step to refine segmentation.
    
    Args:
        image: Original input image, shape (H, W, 3)
        output: Model output, shape (H, W), representing the probability of the foreground
    
    Returns:
        Refined segmentation output (H, W)
    """
    # Ensure image is C-contiguous and convert to uint8
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Convert from float (0-1) to uint8 (0-255)

    h, w = image.shape[:2]
    crf = dcrf.DenseCRF2D(w, h, 2)  # Set num_classes to 2 to internally differentiate foreground vs. background

    # Create unary potential from foreground probability (output)
    foreground_prob = np.stack([1 - output, output], axis=0)  # Create (2, H, W) where first is background prob
    unary = unary_from_softmax(foreground_prob)
    crf.setUnaryEnergy(unary)

    # Add pairwise Gaussian and bilateral potentials to refine segmentation boundaries
    crf.addPairwiseGaussian(sxy=3, compat=3)
    crf.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=10)

    # Perform inference to get refined prediction
    Q = crf.inference(5)
    refined_output = np.argmax(Q, axis=0).reshape((h, w))

    return refined_output
