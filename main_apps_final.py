import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import time
import io
import os
import requests 
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="UpScaling Image AOL ComVis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

ESRGAN_REPO = "yongkytristan/AoL_ComVis"
ESRGAN_FILENAME = "esrgan_lite_full_dynamic.onnx"
SRRESNET_FILENAME = "srresnet.onnx" 

# 🚨 PERUBAHAN PATH SRRESNET LOKAL
# Path model SRResNet sekarang adalah 'srresnet.onnx' di direktori root.
SRRESNET_LOCAL_PATH = SRRESNET_FILENAME

## --- FUNGSI PEMUATAN MODEL DENGAN CACHE TERPISAH ---

@st.cache_resource
def load_esrgan_model_from_hf():
    """Memuat model ESRGAN dari Hugging Face Hub."""
    model_name = "ESRGAN-Lite"
    try:
        model_path = hf_hub_download(repo_id=ESRGAN_REPO, filename=ESRGAN_FILENAME)
        st.success(f"Model '{model_name}' successfully loaded from Hugging Face.")
        
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        return session, input_name, output_name, input_shape

    except Exception as e:
        st.error(f"Failed to load ESRGAN model from Hugging Face.")
        st.exception(e)
        return None, None, None, None

@st.cache_resource
def load_srresnet_model_local():
    """Memuat model SRResNet dari path lokal."""
    model_name = "SRResNet"
    model_path = SRRESNET_LOCAL_PATH 
    
    # Cek keberadaan file lokal
    if not os.path.exists(model_path):
        st.error(f"Model '{model_name}' not found at local path: **{model_path}**")
        st.warning("Pastikan file 'srresnet.onnx' berada di direktori yang sama dengan aplikasi Streamlit ini.")
        # Menggunakan st.info agar alur tidak terhenti, tetapi pengguna tahu ada masalah
        st.info("SRResNet UI tidak akan berfungsi tanpa file model.")
        return None, None, None, None

    try:
        st.success(f"Model '{model_name}' successfully loaded from local path.")
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        return session, input_name, output_name, input_shape
        
    except Exception as e:
        st.error(f"Failed to initialize InferenceSession for local model: {model_name}")
        st.exception(e)
        return None, None, None, None

## --- SRRESNET UTILS (Tidak Berubah) ---

def srresnet_preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_normalized = img_np / 255.0
    # [H, W, C] -> [1, H, W, C]
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def srresnet_postprocess(output_np: np.ndarray) -> Image.Image:
    # [1, H, W, C] -> [H, W, C]
    img_processed = np.squeeze(output_np, axis=0)
    # Denormalize (0-1 -> 0-255) and convert to 8-bit integer
    img_denormalized = (img_processed * 255).astype(np.uint8)
    # Ensure values are within the range [0, 255]
    img_denormalized = np.clip(img_denormalized, 0, 255)
    return Image.fromarray(img_denormalized)

def srresnet_upscale(session, input_name, output_name, lr_image: Image.Image) -> Image.Image: 
    lr_np_input = srresnet_preprocess(lr_image)
    
    # ONNX Inference
    input_feed = {input_name: lr_np_input}
    outputs = session.run([output_name], input_feed)
    sr_np_output = outputs[0]
    
    # Postprocessing
    sr_image = srresnet_postprocess(sr_np_output)
    
    return sr_image

## --- ESRGAN UTILS (Tidak Berubah) ---

def esrgan_upscale(pil_img: Image.Image, session: ort.InferenceSession, input_name: str, output_name: str): 
    # 1. Preprocessing
    # Convert to float32 & normalize 0–1
    lr = np.array(pil_img).astype(np.float32) / 255.0
    # Add batch dim: (H, W, C) -> (1, H, W, C)
    lr_input = lr[None, :, :, :]

    # 2. Inference
    sr = session.run([output_name], {input_name: lr_input})[0]
    
    # 3. Postprocessing
    # Get the first batch, clip & convert to uint8
    sr = np.clip(sr[0], 0.0, 1.0)
    sr_uint8 = (sr * 255.0).astype(np.uint8)

    return Image.fromarray(sr_uint8)

def sharpen(img_pil: Image.Image) -> Image.Image:
    img_uint8 = np.array(img_pil)
    # Commonly used Sharpening Kernel
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32
    )
    # cv2.filter2D applies convolution
    sharpened_np = cv2.filter2D(img_uint8, -1, kernel)
    # Convert back to PIL Image
    return Image.fromarray(sharpened_np)

## --- MAIN APPLICATION LOGIC ---

def display_esrgan_ui(): 
    st.title("🖼️ ESRGAN-Lite Super Resolution")
    st.markdown(
        """
        Aplikasi ini menggunakan model **ESRGAN-Lite** untuk melakukan *upscaling*.
        Model dimuat dari **Hugging Face Hub** (Cloud).
        """
    )

    # 🚨 Panggilan ke fungsi ESRGAN yang terpisah
    session, input_name, output_name, input_shape = load_esrgan_model_from_hf()
    
    if session is None:
        return

    st.sidebar.markdown("#### ESRGAN-Lite Model Info")
    st.sidebar.markdown(f"**Source:** Hugging Face Hub (Cloud)")
    st.sidebar.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your LR (Low-Resolution) image (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        key="esrgan_uploader"
    )

    use_sharpen = st.checkbox("Apply Sharpening after upscaling (optional)", value=False, key="esrgan_sharpen")

    if uploaded_file is not None:
        try:
            lr_pil = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not process the uploaded file. Error: {e}")
            return

        st.subheader("Low-Resolution Image (Input)")
        st.image( 
            lr_pil,
            caption=f"Original Size: {lr_pil.size[0]} × {lr_pil.size[1]} pixels",
            width='content' 
        )

        if st.button("Run UpScale", type="primary", key="esrgan_button"):
            with st.spinner("Processing... Please wait, this depends on image size and your CPU speed."):
                
                t0 = time.time()
                # Run Inference
                sr_pil = esrgan_upscale(lr_pil, session, input_name, output_name)
                infer_time = time.time() - t0
                
                # Apply Sharpening if selected
                sharpen_message = ""
                if use_sharpen:
                    sr_pil = sharpen(sr_pil)
                    sharpen_message = "(with Sharpening)"
                
                st.success(f"Upscaling completed in **{infer_time:.3f} seconds** {sharpen_message} 🎉")
                st.subheader("Super-Resolution Result (Output)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Low Resolution")
                    st.image(lr_pil, width='content')
                with col2:
                    st.markdown("##### Super Resolution")
                    
                    try:
                        upscale_factor = sr_pil.width // lr_pil.width
                        caption_text = f"Result Size: {sr_pil.size[0]} × {sr_pil.size[1]} pixels (Upscale x{upscale_factor})"
                    except ZeroDivisionError:
                           caption_text = f"Result Size: {sr_pil.size[0]} × {sr_pil.size[1]} pixels"
                           
                    st.image(
                        sr_pil,
                        caption=caption_text,
                        width='content'
                    )
                
                buf = io.BytesIO()
                sr_pil.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download SR Result (PNG)",
                    data=buf.getvalue(),
                    file_name="esrgan_sr_output.png",
                    mime="image/png"
                )
    else:
        st.info("Please upload your image file.")


def display_srresnet_ui():
    st.title("🖼️ SRResNet Super Resolution")
    st.markdown(
        """
        Aplikasi ini menggunakan model **SRResNet** untuk melakukan *upscaling*. 
        Model dimuat dari **lokasi file lokal** ('srresnet.onnx').
        """
    )
    
    # 🚨 Panggilan ke fungsi SRResNet yang terpisah
    session, input_name, output_name, input_shape = load_srresnet_model_local()
    
    if session is None:
        return
    
    st.sidebar.markdown("#### SRResNet Model Info")
    st.sidebar.markdown(f"**Source:** Local File (Root Directory)")
    st.sidebar.markdown(f"**Path:** `{SRRESNET_LOCAL_PATH}`")
    st.sidebar.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your LR (Low-Resolution) image (PNG/JPG)", 
        type=["png", "jpg", "jpeg", "webp"],
        key="srresnet_uploader"
    )

    if uploaded_file is not None:
        try:
            lr_image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Could not process the uploaded file. Error: {e}")
            return
        
        st.subheader("Low-Resolution Image (Input)")
        st.image(lr_image, caption=f"Original Size: {lr_image.width}x{lr_image.height}", width='content')

        if st.button("Run Upscale", type="primary", key="srresnet_button"):
            with st.spinner("Processing... Please wait, this depends on image size and your CPU speed."):
                try:
                    t0 = time.time()
                    # Perform upscaling
                    sr_image = srresnet_upscale(session, input_name, output_name, lr_image)
                    infer_time = time.time() - t0
                    
                    st.success(f"Upscaling completed in **{infer_time:.3f} seconds** 🎉")
                    st.subheader("Super-Resolution Result (Output)")
                    
                    col1, col2 = st.columns(2)
                    
                    # Calculate scale factor
                    try:
                        upscale_factor = sr_image.width // lr_image.width
                        caption_text = f"Result Size: {sr_image.width} × {sr_image.height} pixels (Upscale x{upscale_factor})"
                    except ZeroDivisionError:
                        caption_text = f"Result Size: {sr_image.width} × {sr_image.height} pixels"
                        
                    with col1:
                        st.markdown("##### Low Resolution")
                        st.image(lr_image, width='content')
                    
                    with col2:
                        st.markdown("##### Super Resolution")
                        st.image(sr_image, caption=caption_text, width='content')
                    
                    buf = io.BytesIO()
                    sr_image.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download SR Result (PNG)", 
                        data=buf.getvalue(),
                        file_name=f"srresnet_upscaled_{uploaded_file.name}.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
    else:
        st.info("Please upload your image file.")


def main():
    st.sidebar.header("Select Super Resolution Model")

    model_choice = st.sidebar.radio(
        "Select Model:",
        ["ESRGAN-Lite", "SRResNet"],
        index=0, # ESRGAN-Lite as default
    )
    
    if model_choice == "ESRGAN-Lite":
        display_esrgan_ui()
    elif model_choice == "SRResNet":
        display_srresnet_ui()


if __name__ == "__main__":
    main()
