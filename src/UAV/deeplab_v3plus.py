"""
DeepLabV3+ for multi-channel land cover segmentation (UAV cascade copy).

Architecture is IDENTICAL to src/deeplab/deeplab_v3plus.py — kept as a local copy
so the UAV stage is self-contained. The builder is generic in the channel count
and class count (passed as arguments), so the exact same network serves both:
    - the coarse aerial model     (input  7 ch,  8 classes)  -> predict_coarse_tiles.py
    - the fine UAV cascade model  (input 12 ch, refined)     -> train_uav.py

Encoder:  ResNet50, first conv layer adapted to accept N input channels.
ASPP:     Atrous Spatial Pyramid Pooling with rates [6, 12, 18] + global pooling.
Decoder:  Upsample ASPP output, fuse with low-level features, refine, upsample.

Input:  (H, W, C)
Output: (H, W, num_classes), softmax
"""

import tensorflow as tf


def _conv_bn_relu(x, filters, kernel_size=1, dilation_rate=1):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, padding="same",
        dilation_rate=dilation_rate, use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def _build_encoder(input_shape):
    """ResNet50 encoder adapted for an arbitrary number of input channels.

    Pretrained ImageNet weights are not used: the input channel composition
    (UAV R, G, B, nDSM + one-hot coarse-context) is not compatible with natural
    RGB statistics.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    backbone = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )

    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer("conv2_block3_out").output,  # low-level,  stride 4
            backbone.get_layer("conv4_block6_out").output,  # high-level, stride 16
        ],
    )
    low_level, high_level = feature_extractor(inputs)
    return tf.keras.Model(inputs=inputs, outputs=[low_level, high_level], name="encoder")


def _aspp(x, filters=256):
    branches = [
        _conv_bn_relu(x, filters, kernel_size=1),
        _conv_bn_relu(x, filters, kernel_size=3, dilation_rate=6),
        _conv_bn_relu(x, filters, kernel_size=3, dilation_rate=12),
        _conv_bn_relu(x, filters, kernel_size=3, dilation_rate=18),
    ]

    pooled = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    pooled = _conv_bn_relu(pooled, filters, kernel_size=1)
    pooled = tf.keras.layers.UpSampling2D(
        size=(x.shape[1], x.shape[2]), interpolation="bilinear"
    )(pooled)
    branches.append(pooled)

    x = tf.keras.layers.Concatenate()(branches)
    return _conv_bn_relu(x, filters, kernel_size=1)


def build_deeplabv3plus(input_shape=(512, 512, 7), num_classes=8):
    inputs = tf.keras.layers.Input(shape=input_shape)

    encoder = _build_encoder(input_shape)
    low_level, high_level = encoder(inputs)

    x = _aspp(high_level, filters=256)
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    low_level = _conv_bn_relu(low_level, filters=48, kernel_size=1)
    x = tf.keras.layers.Concatenate()([x, low_level])

    x = tf.keras.layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(num_classes, 1)(x)

    # Softmax kept in float32 for numerical stability under mixed precision.
    outputs = tf.keras.layers.Activation("softmax", dtype="float32")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="deeplabv3plus")


if __name__ == "__main__":
    model = build_deeplabv3plus()
    model.summary()
