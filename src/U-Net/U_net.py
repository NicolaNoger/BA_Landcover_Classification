import numpy as np 
import tensorflow as tf

# =============================================================================
# U-NET MODEL DEFINITION
# =============================================================================
# Defines U-Net model with MobileNetV2 encoder for semantic segmentation
# Input:  Images with shape (512, 512, 5) - 5 channels: NIR + RGB + normalized height
# Output: Segmentation masks with shape (512, 512, num_classes)
# =============================================================================


def create_mobilenet_encoder(input_shape=(512, 512, 5), trainable=False):
    """
    Creates MobileNetV2-based encoder for U-Net.
    
    Builds a standalone MobileNetV2 for 5 channels,
    then loads pre-trained weights where possible (first 3 channels).
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        trainable: Whether encoder weights should be trainable
        
    Returns:
        tf.keras.Model: Encoder model that outputs list of feature maps at
                       different scales for skip connections
    """
    # Build MobileNetV2 directly for 5 channels (no weight loading yet)
    inp = tf.keras.layers.Input(shape=input_shape)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None  # No pre-trained weights initially
    )
    base_model.trainable = trainable
    
    # Get intermediate layer outputs for skip connections
    layer_names = [
        'block_1_expand_relu',   # 256x256
        'block_3_expand_relu',   # 128x128
        'block_6_expand_relu',   # 64x64
        'block_13_expand_relu',  # 32x32
        'block_16_project',      # 16x16
    ]
    
    # Extract the output tensors from the base model
    skip_layers = [base_model.get_layer(name).output for name in layer_names]
    skip_layers.append(base_model.output)  # Add bottleneck
    
    # Create encoder model with multiple outputs
    encoder_model = tf.keras.Model(inputs=base_model.input, outputs=skip_layers)
    
    # Apply encoder to our 5-channel input
    skip_outputs = encoder_model(inp)
    
    # Try to load pre-trained weights for first 3 channels
    try:
        print("Loading ImageNet weights for MobileNetV2...")
        pretrained = tf.keras.applications.MobileNetV2(
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Transfer weights layer by layer (skip first conv which has different input channels)
        transferred = 0
        for layer, pretrained_layer in zip(base_model.layers[2:], pretrained.layers[2:]):
            if pretrained_layer.get_weights():
                try:
                    layer.set_weights(pretrained_layer.get_weights())
                    transferred += 1
                except Exception:
                    pass  # Skip layers where shapes don't match
        
        # Handle first conv layer specially (3->5 channels)
        first_conv = base_model.layers[1]  # Usually 'Conv1'
        pretrained_first = pretrained.layers[1]
        
        if first_conv.get_weights() and pretrained_first.get_weights():
            orig_weights = pretrained_first.get_weights()
            kernel = orig_weights[0]  # Shape: (3, 3, 3, 32)
            
            # Map RGB pretrained weights to expected input order: NIR, R, G, B, Height
            rgb_mean = np.mean(kernel, axis=2, keepdims=True)
            red = kernel[:, :, 0:1, :]
            green = kernel[:, :, 1:2, :]
            blue = kernel[:, :, 2:3, :]
            extended_kernel = np.concatenate([rgb_mean, red, green, blue, rgb_mean], axis=2)
            
            if len(orig_weights) > 1:
                first_conv.set_weights([extended_kernel, orig_weights[1]])
            else:
                first_conv.set_weights([extended_kernel])
            transferred += 1
        
        print(f"Successfully transferred weights from {transferred} layers")
        
    except Exception as e:
        print(f"  Warning: Could not load ImageNet weights: {e}")
        print("  Training from scratch (random initialization)")

    return tf.keras.Model(inputs=inp, outputs=skip_outputs)


def upsample_block(x, skip, filters):
    """
    Decoder block with skip connection.
    
    Upsamples feature maps and concatenates with corresponding encoder features
    to recover spatial information lost during downsampling.
    
    Args:
        x: Input feature map from previous decoder layer
        skip: Skip connection from encoder at same scale
        filters: Number of filters for convolutional layers
        
    Returns:
        Upsampled and processed feature map
    """
    # Upsample: Double spatial dimensions (e.g., 16x16 -> 32x32)
    x = tf.keras.layers.Conv2DTranspose(
        filters, 3, strides=2, padding='same'
    )(x)
    
    # Concatenate with skip connection for fine details
    x = tf.keras.layers.Concatenate()([x, skip])
    
    # Process combined features
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    
    return x


def build_unet(input_shape=(512, 512, 5), num_classes=8, use_modified_weights=True):
    """
    Builds complete U-Net architecture with MobileNetV2 encoder.
    
    Architecture:
        - Encoder: MobileNetV2 (pre-trained) with skip connections
        - Bottleneck: Deepest feature representation (16x16)
        - Decoder: Series of upsample blocks with skip connections
        - Output: Pixel-wise classification (512x512xnum_classes)
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of segmentation classes
        use_modified_weights: Whether to use modified MobileNetV2 weights (unused parameter)
        
    Returns:
        tf.keras.Model: Complete U-Net model
    """
    # Create encoder
    encoder = create_mobilenet_encoder(input_shape, trainable=False)
    
    # Build decoder
    inp = tf.keras.layers.Input(shape=input_shape)
    skip_connections = encoder(inp)
    
    # Start from bottleneck
    x = skip_connections[-1]  # 16x16x1280
    
    # 16x16 → 32x32, use skip 3 (32x32x576)
    x = upsample_block(x, skip_connections[3], 512)  # → 32x32
    
    # 32x32 → 64x64, use skip 2 (64x64x192)
    x = upsample_block(x, skip_connections[2], 256)  # → 64x64
    
    # 64x64 → 128x128, use skip 1 (128x128x144)
    x = upsample_block(x, skip_connections[1], 128)  # → 128x128
    
    # 128x128 → 256x256, use skip 0 (256x256x96)
    x = upsample_block(x, skip_connections[0], 64)  # → 256x256
    
    # 256x256 → 512x512, no skip connection
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)  # → 512x512
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    
    # Output layer: Pixel-wise classification
    # softmax: For multi-class segmentation (outputs probability distribution)
    # sigmoid: For binary segmentation (uncomment if needed)
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inp, outputs=outputs)
    
    return model


# Test block to verify model architecture
if __name__ == "__main__":
    print("Building U-Net model...")
    
    # Build model with 8 classes
    model = build_unet(input_shape=(512, 512, 5), num_classes=8)
    
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model.summary()
    
    print("\n" + "="*70)
    print("MODEL INFO")
    print("="*70)
    print(f"Input Shape:  {model.input_shape}")
    print(f"Output Shape: {model.output_shape}")
    print(f"Total Parameters: {model.count_params():,}")
    print("="*70)






