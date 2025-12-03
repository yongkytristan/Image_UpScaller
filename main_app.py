import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import time
import io
import os

# ===============================
# GLOBAL CONFIG
# ===============================
# Pastikan ini adalah perintah Streamlit pertama yang dipanggil
st.set_page_config(
    page_title="UpScalling Image AOL ComVis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Path Model ---
ESRGAN_PATH = "esrgan_lite_full_dynamic.onnx"
SRRESNET_PATH = "srresnet.onnx"

# ===============================
# LOAD ONNX MODEL (CACHED)
# ===============================
@st.cache_resource
def load_onnx_model(model_path):
    """
    Memuat model ONNX ke dalam sesi InferenceSession.
    Menggunakan st.cache_resource agar hanya dimuat sekali.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: Model ONNX tidak ditemukan di path: {model_path}")
        st.stop()
        
    try:
        # Paksa hanya pakai CPUExecutionProvider
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        
        # Ambil nama input & output
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        return session, input_name, output_name, input_shape
    except Exception as e:
        st.error(f"Gagal memuat model ONNX dari path: {model_path}")
        st.exception(e)
        st.stop()
        return None, None, None, None


# ===============================
# SRRESNET UTILS
# ===============================

def srresnet_preprocess(img: Image.Image) -> np.ndarray:
    """Preprocessing untuk model SRResNet: normalisasi 0-1 dan tambah batch dim."""
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_normalized = img_np / 255.0
    # [H, W, C] -> [1, H, W, C]
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def srresnet_postprocess(output_np: np.ndarray) -> Image.Image:
    """Postprocessing untuk model SRResNet: denormalisasi, clip, konversi ke PIL."""
    # [1, H, W, C] -> [H, W, C]
    img_processed = np.squeeze(output_np, axis=0)
    # Denormalisasi (0-1 -> 0-255) dan konversi ke integer 8-bit
    img_denormalized = (img_processed * 255).astype(np.uint8)
    # Pastikan nilai berada dalam rentang [0, 255]
    img_denormalized = np.clip(img_denormalized, 0, 255)
    return Image.fromarray(img_denormalized)

def srresnet_upscale(session, input_name, output_name, lr_image: Image.Image) -> Image.Image:
    """Menjalankan inferensi ONNX untuk SRResNet."""
    
    lr_np_input = srresnet_preprocess(lr_image)
    
    # Inferensi ONNX
    input_feed = {input_name: lr_np_input}
    outputs = session.run([output_name], input_feed)
    sr_np_output = outputs[0]
    
    # Postprocessing
    sr_image = srresnet_postprocess(sr_np_output)
    
    return sr_image

# ===============================
# ESRGAN UTILS
# ===============================

def esrgan_upscale(pil_img: Image.Image, session: ort.InferenceSession, input_name: str, output_name: str):
    """Melakukan inferensi Super Resolution menggunakan ONNX Runtime untuk ESRGAN."""
    
    # 1. Preprocessing
    # Convert ke float32 & normalisasi 0–1
    lr = np.array(pil_img).astype(np.float32) / 255.0
    # Tambah batch dim: (H, W, C) -> (1, H, W, C)
    lr_input = lr[None, :, :, :]

    # 2. Inference
    sr = session.run([output_name], {input_name: lr_input})[0]
    
    # 3. Postprocessing
    # Ambil batch pertama, clip & convert ke uint8
    sr = np.clip(sr[0], 0.0, 1.0)
    sr_uint8 = (sr * 255.0).astype(np.uint8)

    return Image.fromarray(sr_uint8)

def sharpen(img_pil: Image.Image) -> Image.Image:
    """Menerapkan filter sharpening sederhana menggunakan kernel Laplacian."""
    # Konversi ke array OpenCV
    img_uint8 = np.array(img_pil)
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32
    )
    # cv2.filter2D menerapkan konvolusi
    sharpened_np = cv2.filter2D(img_uint8, -1, kernel)
    # Konversi kembali ke PIL Image
    return Image.fromarray(sharpened_np)

# ===============================
# MAIN APPLICATION LOGIC
# ===============================

def display_esrgan_ui():
    """Tampilkan UI dan logika untuk model ESRGAN-Lite."""
    
    st.title("🖼️ ESRGAN-Lite Super Resolution")
    st.markdown(
        """
        Aplikasi ini menggunakan model **ESRGAN-Lite** untuk melakukan *upscaling* gambar. Model ini umumnya memberikan kualitas visual yang lebih baik.
        """
    )

    # Load Model (dengan caching)
    session, input_name, output_name, input_shape = load_onnx_model(ESRGAN_PATH)
    
    # Sidebar Info
    st.sidebar.markdown("#### Info Model ESRGAN-Lite")
    st.sidebar.markdown(f"**Model Path:** `{ESRGAN_PATH}`")
    st.sidebar.markdown("---")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload gambar LR (Low-Resolution) Anda (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        key="esrgan_uploader"
    )

    # Opsi Tambahan
    use_sharpen = st.checkbox("Terapkan Sharpening setelah upscaling (opsional)", value=False, key="esrgan_sharpen")

    if uploaded_file is not None:
        try:
            lr_pil = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Tidak dapat memproses file yang diunggah. Error: {e}")
            return

        st.subheader("Gambar Low-Resolution (Input)")
        st.image(
            lr_pil,
            caption=f"Ukuran Asli: {lr_pil.size[0]} × {lr_pil.size[1]} pixels",
            use_container_width=True
        )

        if st.button("Jalankan UpScale", type="primary", key="esrgan_button"):
            with st.spinner("Memproses... Harap tunggu, ini bergantung pada ukuran gambar dan kecepatan CPU Anda."):
                
                t0 = time.time()
                # Jalankan Inferensi
                sr_pil = esrgan_upscale(lr_pil, session, input_name, output_name)
                infer_time = time.time() - t0
                
                # Terapkan Sharpening jika dipilih
                sharpen_message = ""
                if use_sharpen:
                    sr_pil = sharpen(sr_pil)
                    sharpen_message = "(dengan Sharpening)"
                
                # Tampilkan Hasil
                st.success(f"Upscaling selesai dalam **{infer_time:.3f} detik** {sharpen_message} 🎉")
                st.subheader("Hasil Super-Resolution (Output)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Low Resolution")
                    st.image(lr_pil, use_container_width=True)
                with col2:
                    st.markdown("##### Super Resolution")
                    st.image(
                        sr_pil,
                        caption=f"Ukuran Hasil: {sr_pil.size[0]} × {sr_pil.size[1]} pixels",
                        use_container_width=True
                    )
                
                # Tombol Download
                buf = io.BytesIO()
                sr_pil.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Hasil SR (PNG)",
                    data=buf.getvalue(),
                    file_name="esrgan_sr_output.png",
                    mime="image/png"
                )
    else:
        st.info("Silakan upload file gambar Anda. Pastikan file model ONNX tersedia di direktori yang sama.")


def display_srresnet_ui():
    """Tampilkan UI dan logika untuk model SRResNet."""
    
    st.title("🖼️ SRResNet Super Resolution")
    st.markdown(
        """
        Aplikasi ini menggunakan model **SRResNet** untuk melakukan *upscaling* gambar. Model ini lebih cepat tetapi hasilnya mungkin tidak se-detail ESRGAN.
        """
    )
    
    # Load Model (dengan caching)
    session, input_name, output_name, input_shape = load_onnx_model(SRRESNET_PATH)
    
    # --- INFO SIDEBAR (Disamakan dengan ESRGAN) ---
    st.sidebar.markdown("#### Info Model SRResNet")
    st.sidebar.markdown(f"**Model Path:** `{SRRESNET_PATH}`")
    st.sidebar.markdown("---")
    # -----------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload gambar LR (Low-Resolution) Anda (PNG/JPG)", # Label disamakan
        type=["png", "jpg", "jpeg", "webp"],
        key="srresnet_uploader"
    )

    # SRResNet tidak memiliki opsi sharpening di kode aslinya, jadi kita lewati langkah 2.

    if uploaded_file is not None:
        # Memuat gambar yang diunggah
        lr_image = Image.open(uploaded_file)

        st.subheader("Gambar Low-Resolution (Input)") # Header disamakan
        st.image(lr_image, caption=f"Ukuran Asli: {lr_image.width}x{lr_image.height}", use_container_width=True)
        # st.write(f"Resolusi Asli: {lr_image.width}x{lr_image.height}") # Dipindah ke caption/bagian hasil

        if st.button("Jalankan Upscale", type="primary", key="srresnet_button"): # Tombol disamakan
            with st.spinner("Memproses... Harap tunggu, ini bergantung pada ukuran gambar dan kecepatan CPU Anda."):
                try:
                    t0 = time.time()
                    # Lakukan upscaling
                    sr_image = srresnet_upscale(session, input_name, output_name, lr_image)
                    infer_time = time.time() - t0
                    
                    st.success(f"Upscaling selesai dalam **{infer_time:.3f} detik** 🎉")
                    st.subheader("Hasil Super-Resolution (Output)") # Header disamakan
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Low Resolution") # Subheader disamakan
                        st.image(lr_image, caption=f"Low Resolution ({lr_image.width}x{lr_image.height})", use_container_width=True)
                    
                    with col2:
                        st.markdown("##### Super Resolution") # Subheader disamakan
                        try:
                            upscale_factor = sr_image.width // lr_image.width
                            caption_text = f"Ukuran Hasil: {sr_image.width} × {sr_image.height} pixels (Upscale x{upscale_factor})"
                        except ZeroDivisionError:
                            caption_text = f"Ukuran Hasil: {sr_image.width} × {sr_image.height} pixels"

                        st.image(sr_image, caption=caption_text, use_container_width=True)
                    
                    
                    # Opsi Download
                    buf = io.BytesIO()
                    sr_image.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Hasil SR (PNG)", # Label disamakan
                        data=buf.getvalue(),
                        file_name=f"srresnet_upscaled_{uploaded_file.name}.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menjalankan prediksi: {e}")
    else:
        st.info("Silakan upload file gambar Anda. Pastikan file model ONNX tersedia di direktori yang sama.")


def main():
    
    # ===============================
    # Sidebar Navigation
    # ===============================
    st.sidebar.header("Pilih Model Super Resolution")
    
    # Menggunakan st.radio untuk navigasi
    model_choice = st.sidebar.radio(
        "Pilih Model:",
        ["ESRGAN-Lite", "SRResNet"],
        index=0, # ESRGAN-Lite sebagai default
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Pastikan file ONNX yang sesuai tersedia di direktori yang sama.")
    
    # Tampilkan UI sesuai pilihan
    if model_choice == "ESRGAN-Lite":
        display_esrgan_ui()
    elif model_choice == "SRResNet":
        display_srresnet_ui()


if __name__ == "__main__":
    main()