import tensorflow as tf
from tensorflow.keras.layers import PReLU, Add
import numpy as np
import math
from IPython.display import FileLink, display, Javascript
import os

if not hasattr(np, 'object'):
    np.object = object

import tf2onnx 


class PixelShuffleLayer(tf.keras.layers.Layer):
    """Custom layer untuk operasi Upsampling/Pixel Shuffling (tf.nn.depth_to_space)."""
    def __init__(self, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, x):
        return tf.nn.depth_to_space(x, self.scale)
    
    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale})
        return config

MODEL_PATH = "srresnet_final.keras"
ONNX_OUTPUT_PATH = "srresnet.onnx"
SCALE_FACTOR = 4

CUSTOM_OBJECTS = {
    'PixelShuffleLayer': PixelShuffleLayer,
    'PReLU': PReLU 
}

def download_file(path):
    """Membuat tautan download untuk file yang diberikan di lingkungan notebook."""
    if not os.path.exists(path):
        print(f"❌ ERROR: File {path} tidak ditemukan untuk didownload.")
        return

    try:
        file_size = os.path.getsize(path)
        js_script = f"""
        (async function() {{
          const filename = "{os.path.basename(path)}";
          const filesize = {file_size};
          const url = "/files/{path}"; 
          
          const a = document.createElement('a');
          a.href = url;
          a.download = filename;
          
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          console.log(`Download triggered for ${{filename}} (${{filesize}} bytes)`);
        }})();
        """
        display(Javascript(js_script))
        print(f"✅ Download file '{os.path.basename(path)}' telah dipicu secara otomatis.")
        
    except Exception as e:
        print(f"⚠️ Gagal memicu download otomatis: {e}. Menampilkan tautan manual...")
        display(FileLink(path))


def convert_keras_to_onnx(model_path, output_path, custom_objects):
    print(f"1. Memuat model Keras dari: {model_path}")
    
    try:
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
    except Exception as e:
        print(f"❌ ERROR: Gagal memuat model. Pastikan file '{model_path}' tersedia.")
        print(f"Detail error: {e}")
        return

    print("2. Model Keras berhasil dimuat. Siap dikonversi ke ONNX.")
    
    input_signature = [
        tf.TensorSpec(
            [None, None, None, 3], # [Batch, Height, Width, Channels]
            tf.float32, 
            name='input_image'
        )
    ]

    try:
        onnx_model, external_tensor_storage = tf2onnx.convert.from_keras(
            model, 
            input_signature=input_signature, 
            opset=13,
            output_path=output_path,
        )
        
        print(f"3. Konversi selesai! 🎉")
        print(f"   Model ONNX tersimpan di: {output_path}")
        print(f"   Ukuran model ONNX: {len(onnx_model.graph.node)} nodes")
        
        print("\n4. Mempersiapkan download file ONNX...")
        download_file(output_path)
        
    except Exception as e:
        print(f"❌ ERROR: Gagal dalam konversi ke ONNX.")
        print(f"Detail error: {e}")


if __name__ == "__main__":
    convert_keras_to_onnx(MODEL_PATH, ONNX_OUTPUT_PATH, CUSTOM_OBJECTS)