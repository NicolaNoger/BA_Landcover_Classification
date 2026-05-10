import numpy as np
import tensorflow as tf

# =============================================================================
# DEEPLABV3+ MODEL DEFINITION
# =============================================================================
# Encoder:  ResNet50 backbone adapted for N input channels
#           Low-level features  at stride 4  (conv2_block3_out, 256ch)
#           High-level features at stride 16 (conv4_block6_out, 1024ch)
# ASPP:     Atrous Spatial Pyramid Pooling with rates [6, 12, 18] + GAP
# Decoder:  Bilinear 4× upsample → concat low-level → 2× SepConv → 4× upsample → softmax
#
# Input:  (512, 512, C)   C = 7 channels (NIR R G B nDSM + 2)
# Output: (512, 512, num_classes)
#
# Why DeepLabV3+ over U-Net for this task:
#   - ASPP captures multi-scale context without exploding memory
#   - Atrous convolutions preserve spatial resolution without pooling
#   - More stable training on scenes with large homogeneous regions
#     (Cropland, Grassland) because global context is available everywhere
# =============================================================================


# -----------------------------------------------------------------------------
# BACKBONE
# -----------------------------------------------------------------------------

def _build_resnet_encoder(input_shape: tuple, trainable: bool = False):
    """
    ResNet50 backbone adapted for arbitrary channel count via pretrained
    weight transfer from the 3-channel ImageNet model.

    Weight transfer strategy for the first conv (7×7, 3→C channels):
      - R, G, B channels: copied directly from pretrained weights
      - NIR channel (idx 0): initialized from mean of RGB weights
      - All remaining channels: initialized from mean of RGB weights

    Returns a Keras Model with two outputs:
        [0]  low_level  – stride-4  features, shape (..., H/4,  W/4,  256)
        [1]  high_level – stride-16 features, shape (..., H/16, W/16, 1024)
    """
    inp = tf.keras.layers.Input(shape=input_shape)
    n_channels = input_shape[-1]

    # Build ResNet50 without pretrained weights for N channels
    backbone = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )
    backbone.trainable = trainable

    # Feature extraction layers
    low_level_layer  = backbone.get_layer("conv2_block3_out")   # stride 4
    high_level_layer = backbone.get_layer("conv4_block6_out")   # stride 16

    feature_model = tf.keras.Model(
        inputs  = backbone.input,
        outputs = [low_level_layer.output, high_level_layer.output],
    )

    # Apply to our input
    low_level, high_level = feature_model(inp)

    # Transfer pretrained ImageNet weights
    try:
        print("Loading ImageNet weights for ResNet50 backbone...")
        pretrained = tf.keras.applications.ResNet50(
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=False,
            weights="imagenet",
        )

        transferred = 0
        # Transfer all layers except the very first conv (different channel count)
        for layer, pt_layer in zip(backbone.layers[2:], pretrained.layers[2:]):
            if layer.name == pt_layer.name and pt_layer.get_weights():
                try:
                    layer.set_weights(pt_layer.get_weights())
                    transferred += 1
                except Exception:
                    pass

        # First conv: (7, 7, 3, 64) → (7, 7, C, 64)
        first_conv    = backbone.layers[1]
        pt_first_conv = pretrained.layers[1]
        if first_conv.get_weights() and pt_first_conv.get_weights():
            pt_kernel = pt_first_conv.get_weights()[0]   # (7, 7, 3, 64)
            rgb_mean  = pt_kernel.mean(axis=2, keepdims=True)  # (7, 7, 1, 64)

            # Build (7, 7, C, 64) kernel: slot RGB channels in, fill rest with mean
            new_kernel = np.repeat(rgb_mean, n_channels, axis=2).astype(np.float32)
            # Overwrite R, G, B slots with actual pretrained weights
            # Assumed channel order: NIR=0, R=1, G=2, B=3, rest=4+
            for src_c, dst_c in [(0, 1), (1, 2), (2, 3)]:
                if dst_c < n_channels:
                    new_kernel[:, :, dst_c, :] = pt_kernel[:, :, src_c, :]

            weights = [new_kernel]
            if len(pt_first_conv.get_weights()) > 1:
                weights.append(pt_first_conv.get_weights()[1])
            first_conv.set_weights(weights)
            transferred += 1

        print(f"  Transferred weights from {transferred} layers")

    except Exception as exc:
        print(f"  Warning: ImageNet weight transfer failed – training from scratch. ({exc})")

    return tf.keras.Model(inputs=inp, outputs=[low_level, high_level])


# -----------------------------------------------------------------------------
# ASPP MODULE
# -----------------------------------------------------------------------------

def _aspp(x: tf.Tensor, filters: int = 256) -> tf.Tensor:
    """
    Atrous Spatial Pyramid Pooling.

    Applies parallel dilated convolutions at rates [1, 6, 12, 18] plus
    global average pooling, concatenates all branches, then projects to
    `filters` channels.

    Args:
        x:       Input tensor  (B, H, W, C)
        filters: Output channels for every branch and the final projection

    Returns:
        Tensor (B, H, W, filters)
    """
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]

    def _conv_bn_relu(inp, f, k=1, rate=1):
        y = tf.keras.layers.Conv2D(
            f, k, padding="same", dilation_rate=rate, use_bias=False
        )(inp)
        y = tf.keras.layers.BatchNormalization()(y)
        return tf.keras.layers.ReLU()(y)

    # 1×1 conv
    b0 = _conv_bn_relu(x, filters, k=1, rate=1)

    # Dilated 3×3 convs
    b1 = _conv_bn_relu(x, filters, k=3, rate=6)
    b2 = _conv_bn_relu(x, filters, k=3, rate=12)
    b3 = _conv_bn_relu(x, filters, k=3, rate=18)

    # Global average pooling branch (captures image-level context)
    b4 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    b4 = _conv_bn_relu(b4, filters, k=1)
    b4 = tf.keras.layers.UpSampling2D(size=(x.shape[1] if x.shape[1] is not None else 32, x.shape[2] if x.shape[2] is not None else 32), interpolation="bilinear")(b4)

    out = tf.keras.layers.Concatenate()([b0, b1, b2, b3, b4])
    out = _conv_bn_relu(out, filters, k=1)
    return out


# -----------------------------------------------------------------------------
# FULL MODEL
# -----------------------------------------------------------------------------

def build_deeplabv3plus(
    input_shape: tuple = (512, 512, 7),
    num_classes: int   = 8,
) -> tf.keras.Model:
    """
    Builds DeepLabV3+ for semantic segmentation.

    Architecture summary:
        Input (512×512×C)
          ↓ ResNet50 backbone
        Low-level  (128×128×256)   High-level (32×32×1024)
                                     ↓ ASPP
                                   (32×32×256)
                                     ↓ 4× bilinear upsample
                                   (128×128×256)
          ↓ 1×1 conv (48ch)          ↓
        (128×128×48) ─── concat ──► (128×128×304)
                                     ↓ SepConv 3×3
                                     ↓ SepConv 3×3
                                   (128×128×256)
                                     ↓ 4× bilinear upsample
                                   (512×512×256)
                                     ↓ 1×1 conv + softmax
                                   (512×512×num_classes)

    Args:
        input_shape: (H, W, C) – must match dataloader output
        num_classes: number of segmentation classes

    Returns:
        tf.keras.Model
    """
    inp = tf.keras.layers.Input(shape=input_shape)

    # Backbone
    encoder  = _build_resnet_encoder(input_shape, trainable=False)
    low_feat, high_feat = encoder(inp)   # (B,128,128,256), (B,32,32,1024)

    # ASPP on high-level features
    aspp_out = _aspp(high_feat, filters=256)          # (B, 32, 32, 256)

    # Upsample ASPP output to low-level resolution (×4)
    # The output of ResNet50 high_feat is 32x32, low_feat is 128x128. So we need UpSampling2D(size=(4,4))
    aspp_up = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(aspp_out)

    # Project low-level features to 48 channels (paper recommendation)
    low_proj = tf.keras.layers.Conv2D(48, 1, use_bias=False)(low_feat)
    low_proj = tf.keras.layers.BatchNormalization()(low_proj)
    low_proj = tf.keras.layers.ReLU()(low_proj)        # (B, 128, 128, 48)

    # Concat and refine
    x = tf.keras.layers.Concatenate()([aspp_up, low_proj])   # (B, 128, 128, 304)

    x = tf.keras.layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)                      # (B, 128, 128, 256)

    # Final upsample to full resolution (×4 → 512×512)
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    # Classification head
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(x)

    return tf.keras.Model(inputs=inp, outputs=outputs)


# -----------------------------------------------------------------------------
# QUICK SANITY CHECK
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_deeplabv3plus(input_shape=(512, 512, 7), num_classes=8)
    model.summary()
    print(f"\nInput:      {model.input_shape}")
    print(f"Output:     {model.output_shape}")
    print(f"Parameters: {model.count_params():,}")
