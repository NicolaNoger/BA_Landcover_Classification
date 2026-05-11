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
# =============================================================================

# -----------------------------------------------------------------------------
# BACKBONE
# -----------------------------------------------------------------------------

def _build_resnet_encoder(input_shape: tuple, weights_path: str = None):
    """
    ResNet50 backbone for arbitrary channel count.
    """
    n_channels = input_shape[-1]

    cached_weights: dict = {}
    try:
        # --- ANPASSUNG: Lade Gewichte von lokalem Pfad statt aus dem Internet ---
        if weights_path:
            print(f"Loading local weights from {weights_path} for ResNet50 backbone...")
        else:
            print("Loading weights from scratch (no path provided)...")
            
        _tmp = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None, # "imagenet" entfernt, um Download-Fehler zu vermeiden
        )
        if weights_path:
            _tmp.load_weights(weights_path)
        # ------------------------------------------------------------------------

        for layer in _tmp.layers:
            w = layer.get_weights()
            if w:
                cached_weights[layer.name] = w
        del _tmp

        # Adapt first conv: (7,7,3,64) → (7,7,C,64)
        if "conv1_conv" in cached_weights:
            pt_k  = cached_weights["conv1_conv"][0]              # (7,7,3,64)
            mean_k = pt_k.mean(axis=2, keepdims=True)             # (7,7,1,64)
            new_k  = np.repeat(mean_k, n_channels, axis=2).astype(np.float32)
            for src_c, dst_c in [(0, 1), (1, 2), (2, 3)]:        # R→1, G→2, B→3
                if dst_c < n_channels:
                    new_k[:, :, dst_c, :] = pt_k[:, :, src_c, :]
            adapted = [new_k]
            if len(cached_weights["conv1_conv"]) > 1:
                adapted.append(cached_weights["conv1_conv"][1])
            cached_weights["conv1_conv"] = adapted

        print(f"  Cached weights from {len(cached_weights)} layers")

    except Exception as exc:
        print(f"  Warning: could not load weights ({exc}). Training from scratch.")

    inp = tf.keras.layers.Input(shape=input_shape)
    backbone = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=None,
    )
    backbone.trainable = True

    if cached_weights:
        transferred = 0
        for layer in backbone.layers:
            if layer.name in cached_weights:
                try:
                    layer.set_weights(cached_weights[layer.name])
                    transferred += 1
                except Exception:
                    pass  
        print(f"  Applied weights to {transferred} / {len(backbone.layers)} backbone layers")

    low_level_layer  = backbone.get_layer("conv2_block3_out")   
    high_level_layer = backbone.get_layer("conv4_block6_out")   

    feature_model = tf.keras.Model(
        inputs  = backbone.input,
        outputs = [low_level_layer.output, high_level_layer.output],
    )
    low_level, high_level = feature_model(inp)

    return tf.keras.Model(inputs=inp, outputs=[low_level, high_level])


# -----------------------------------------------------------------------------
# ASPP MODULE
# -----------------------------------------------------------------------------

def _aspp(x: tf.Tensor, filters: int = 256) -> tf.Tensor:
    def _conv_bn_relu(inp, f, k=1, rate=1):
        y = tf.keras.layers.Conv2D(
            f, k, padding="same", dilation_rate=rate, use_bias=False
        )(inp)
        y = tf.keras.layers.BatchNormalization()(y)
        return tf.keras.layers.ReLU()(y)

    b0 = _conv_bn_relu(x, filters, k=1, rate=1)
    b1 = _conv_bn_relu(x, filters, k=3, rate=6)
    b2 = _conv_bn_relu(x, filters, k=3, rate=12)
    b3 = _conv_bn_relu(x, filters, k=3, rate=18)

    b4 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    b4 = _conv_bn_relu(b4, filters, k=1)
    b4 = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation="bilinear")(b4)

    out = tf.keras.layers.Concatenate()([b0, b1, b2, b3, b4])
    out = _conv_bn_relu(out, filters, k=1)
    return out


# -----------------------------------------------------------------------------
# FULL MODEL
# -----------------------------------------------------------------------------

def build_deeplabv3plus(
    input_shape: tuple = (512, 512, 7),
    num_classes: int   = 8,
    weights_path: str  = None, # --- ANPASSUNG: Parameter hinzugefügt ---
) -> tf.keras.Model:

    inp = tf.keras.layers.Input(shape=input_shape)

    # Backbone (Übergabe des weights_path)
    encoder  = _build_resnet_encoder(input_shape, weights_path)
    low_feat, high_feat = encoder(inp)   

    # ASPP on high-level features
    aspp_out = _aspp(high_feat, filters=256)          

    aspp_up = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(aspp_out)

    low_proj = tf.keras.layers.Conv2D(48, 1, use_bias=False)(low_feat)
    low_proj = tf.keras.layers.BatchNormalization()(low_proj)
    low_proj = tf.keras.layers.ReLU()(low_proj)        

    # Concat and refine
    x = tf.keras.layers.Concatenate()([aspp_up, low_proj])   

    x = tf.keras.layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)                      

    # Final upsample to full resolution
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    # --- ANPASSUNG: Zwingend float32 für den Softmax Output, wichtig für AMP ---
    x = tf.keras.layers.Conv2D(num_classes, 1)(x)
    outputs = tf.keras.layers.Activation("softmax", dtype="float32")(x)
    # ---------------------------------------------------------------------------

    return tf.keras.Model(inputs=inp, outputs=outputs)

if __name__ == "__main__":
    model = build_deeplabv3plus(input_shape=(512, 512, 7), num_classes=8)
    model.summary()