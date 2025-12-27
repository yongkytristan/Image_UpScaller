import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, Add
from tensorflow.keras.models import load_model
import tf2onnx
import onnx

# ==============================
# CUSTOM LAYERS (HARUS ADA)
# ==============================

@tf.keras.utils.register_keras_serializable()
class RDB(Layer):
    def __init__(self, filters=64, gc=32, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.gc = gc

    def build(self, input_shape):
        self.c1 = Conv2D(self.gc, 3, padding='same')
        self.c2 = Conv2D(self.gc, 3, padding='same')
        self.c3 = Conv2D(self.gc, 3, padding='same')
        self.c4 = Conv2D(self.gc, 3, padding='same')
        self.c5 = Conv2D(self.filters, 3, padding='same')
        self.act = LeakyReLU(0.2)
        super().build(input_shape)

    def call(self, x):
        c1 = self.act(self.c1(x))
        c2 = self.act(self.c2(tf.concat([x, c1], axis=-1)))
        c3 = self.act(self.c3(tf.concat([x, c1, c2], axis=-1)))
        c4 = self.act(self.c4(tf.concat([x, c1, c2, c3], axis=-1)))
        c5 = self.c5(tf.concat([x, c1, c2, c3, c4], axis=-1))
        return x + 0.2 * c5


@tf.keras.utils.register_keras_serializable()
class RRDB(Layer):
    def __init__(self, filters=64, gc=32, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.gc = gc

    def build(self, input_shape):
        self.rdb1 = RDB(self.filters, self.gc)
        self.rdb2 = RDB(self.filters, self.gc)
        self.rdb3 = RDB(self.filters, self.gc)
        super().build(input_shape)

    def call(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


@tf.keras.utils.register_keras_serializable()
class PixelShuffle(Layer):
    def __init__(self, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, x):
        return tf.nn.depth_to_space(x, self.scale)


# ==============================
# LOAD MODEL
# ==============================

KERAS_PATH = "esrgan_lite_full.keras"

model = load_model(
    KERAS_PATH,
    compile=False,
    custom_objects={"RDB": RDB, "RRDB": RRDB, "PixelShuffle": PixelShuffle}
)

# ==============================
# SET DYNAMIC INPUT (H,W bebas)
# ==============================

input_signature = [
    tf.TensorSpec(
        [1, None, None, 3],   # <--- dynamic height & width
        tf.float32,
        name="input"
    )
]

# ==============================
# CONVERT TO ONNX
# ==============================

onnx_model_path = "esrgan_lite_full_dynamic.onnx"

print("Converting to dynamic ONNX...")

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path=onnx_model_path
)

print("Saved:", onnx_model_path)

# Verify
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

print("ONNX dynamic model is valid!")
