print("Prediction System for Newly Uploaded Data")

"""Streamlit Content Moderation App
- Predicts single images and videos using a trained Keras model
- Upload image or video, see prediction, confidence, and detailed video frame analysis
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import json
import pandas as pd
from io import BytesIO
import time

# Model configuration
MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), r"C:\Users\LENOVO\.vscode\Raxs-AI\models\best_model.h5"),
    os.path.join(os.path.dirname(__file__), r"C:\Users\LENOVO\.vscode\Raxs-AI\models\final_model.keras"),
    "../models/final_model.keras",
    "../models/best_model.h5"
]


IMG_HEIGHT = 224
IMG_WIDTH = 224

@st.cache_resource
def load_moderation_model():
    """Load the trained model with multiple fallback paths"""
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                st.success(f"✅ Model loaded successfully from: {model_path}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Failed to load from {model_path}: {e}")
                continue
    
    st.error("❌ No trained model found. Please train the model first.")
    st.info("📁 Expected model locations:")
    for path in MODEL_PATHS:
        st.write(f"  - {path}")
    return None

def preprocess_image_array(img_array):
    """Preprocess image for model prediction"""
    img = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model, img_array):
    """Predict single image and return detailed results"""
    x = preprocess_image_array(img_array)
    pred = float(model.predict(x, verbose=0)[0][0])
    label = "Nude" if pred > 0.5 else "Safe"
    confidence = pred if pred > 0.5 else 1.0 - pred
    
    return {
        "prediction": pred, 
        "class": label, 
        "confidence": confidence, 
        "is_nude": pred > 0.5,
        "safe_for_display": pred <= 0.5,
        "message": "🚫 UNSAFE - Content violates guidelines" if pred > 0.5 else "✅ SAFE - Appropriate content"
    }

def analyze_video(model, video_path, frame_interval=10, threshold=0.2, progress_bar=None, status_text=None):
    """Analyze video file frame by frame"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    analyzed = 0
    nude_frames = 0
    frame_results = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        analyzed += 1
        
        # Update progress
        if progress_bar and total_frames:
            progress_bar.progress(min(1.0, frame_idx / total_frames))
        if status_text:
            status_text.text(f"Analyzing frame {frame_idx}/{total_frames}...")
        
        # Predict frame
        res = predict_image(model, frame)
        frame_results.append({
            "frame": frame_idx,
            "timestamp": frame_idx / fps if fps > 0 else 0,
            **res
        })
        
        if res['is_nude']:
            nude_frames += 1

    cap.release()

    # Determine overall video classification
    nude_ratio = nude_frames / max(analyzed, 1)
    is_video_nude = nude_ratio > threshold

    return {
        "class": "Nude" if is_video_nude else "Safe",
        "nude_ratio": float(nude_ratio),
        "nude_frames": nude_frames,
        "total_frames_analyzed": analyzed,
        "total_frames": total_frames,
        "is_nude": bool(is_video_nude),
        "safe_for_display": not is_video_nude,
        "duration_seconds": float(duration),
        "fps": float(fps),
        "frame_predictions": frame_results,
        "message": "🚫 UNSAFE - Video contains inappropriate content" if is_video_nude else "✅ SAFE - Video is appropriate for display"
    }

def bytes_to_image(file_bytes):
    """Convert uploaded file bytes to image array"""
    img = Image.open(BytesIO(file_bytes)).convert('RGB')
    return np.array(img)

def display_image_results(res, img_array):
    """Display image prediction results"""
    st.subheader("📊 Prediction Results")
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Classification with color coding
        if res['is_nude']:
            st.error(f"**Classification:** {res['class']} 🚫")
        else:
            st.success(f"**Classification:** {res['class']} ✅")
        
        st.metric("Confidence", f"{res['confidence']:.2%}")
        st.write(f"**Raw Score:** {res['prediction']:.4f}")
        st.write(f"**Safe for Display:** {res['safe_for_display']}")
    
    with col2:
        st.image(img_array, caption="Uploaded Image", use_column_width=True)
    
    st.info(res['message'])

def display_video_results(summary):
    """Display video analysis results"""
    st.subheader("📊 Video Analysis Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if summary['is_nude']:
            st.error(f"**Classification:** {summary['class']} 🚫")
        else:
            st.success(f"**Classification:** {summary['class']} ✅")
        
        st.metric("Nude Frame Ratio", f"{summary['nude_ratio']:.2%}")
    
    with col2:
        st.metric("Frames Analyzed", summary['total_frames_analyzed'])
        st.metric("Total Frames", summary['total_frames'])
    
    with col3:
        st.metric("Duration", f"{summary['duration_seconds']:.1f}s")
        st.metric("Nude Frames", summary['nude_frames'])
    
    st.info(summary['message'])
    
    # Frame analysis details
    if summary['frame_predictions']:
        st.subheader("📋 Frame-by-Frame Analysis")
        
        # Create dataframe for better display
        df = pd.DataFrame(summary['frame_predictions'])
        
        # Add visual indicators
        df['Status'] = df['is_nude'].map({True: '🚫 Nude', False: '✅ Safe'})
        df['Timestamp'] = df['timestamp'].apply(lambda x: f"{x:.1f}s")
        
        # Display relevant columns
        display_df = df[['frame', 'Timestamp', 'Status', 'confidence', 'prediction']]
        display_df.columns = ['Frame', 'Timestamp', 'Status', 'Confidence', 'Raw Score']
        
        st.dataframe(display_df.head(20), use_container_width=True)
        
        # Show some statistics
        st.write(f"**Analysis Settings:** Frame interval used, threshold applied")
        
        # Download full data
        if len(df) > 20:
            st.info(f"Showing first 20 of {len(df)} analyzed frames. Download full report below.")

# --- Streamlit UI ---
st.set_page_config(
    page_title="Content Moderation AI", 
    layout='wide',
    page_icon="🔎"
)

st.title("🔎 AI Content Moderation System")
st.markdown("Upload images or videos to automatically detect inappropriate content using deep learning.")

# Initialize model
model = load_moderation_model()

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 Upload & Settings")
    
    # File type selection
    input_type = st.radio(
        "Choose content type:",
        ["Image", "Video"],
        help="Select whether you're uploading an image or video"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi', 'webm'],
        help="Supported formats: Images (JPG, PNG), Videos (MP4, MOV, AVI)"
    )
    
    # Video-specific settings
    if input_type == "Video":
        st.subheader("🎥 Video Analysis Settings")
        frame_interval = st.slider(
            "Frame Analysis Interval",
            min_value=1,
            max_value=30,
            value=10,
            help="Analyze every Nth frame (higher = faster processing)"
        )
        threshold = st.slider(
            "Nude Content Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Minimum ratio of nude frames to classify video as inappropriate"
        )
    
    st.markdown("---")
    st.header("⚙️ Options")
    show_detailed_analysis = st.checkbox("Show detailed frame analysis", value=True)
    enable_json_export = st.checkbox("Enable JSON report download", value=True)

with col2:
    st.header("📊 Results")
    results_container = st.container()

# Process uploaded file
if uploaded_file is not None:
    if model is None:
        st.error("❌ Model not loaded. Please ensure the model files exist and try again.")
    else:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
        
        try:
            if input_type == "Image":
                with st.spinner("🖼️ Analyzing image..."):
                    # Process image
                    img_array = bytes_to_image(open(temp_path, 'rb').read())
                    result = predict_image(model, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                    
                    # Display results
                    with results_container:
                        display_image_results(result, img_array)
                        
                        # JSON export
                        if enable_json_export:
                            st.download_button(
                                "📥 Download JSON Report",
                                json.dumps(result, indent=2),
                                file_name=f"image_analysis_{int(time.time())}.json",
                                mime="application/json"
                            )
            
            else:  # Video
                with st.spinner("🎥 Processing video... This may take a while depending on video length."):
                    # Create progress elements
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Analyze video
                    summary = analyze_video(
                        model, 
                        temp_path, 
                        frame_interval=frame_interval, 
                        threshold=threshold,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    with results_container:
                        if 'error' in summary:
                            st.error(f"❌ {summary['error']}")
                        else:
                            display_video_results(summary)
                            
                            # JSON export
                            if enable_json_export:
                                st.download_button(
                                    "📥 Download Full Analysis Report",
                                    json.dumps(summary, indent=2),
                                    file_name=f"video_analysis_{int(time.time())}.json",
                                    mime="application/json"
                                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
        
        finally:
            # Cleanup temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

else:
    with col2:
        st.info("👆 Please upload an image or video file to get started.")
        
        # Show supported formats
        st.subheader("📋 Supported Formats")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Images:**")
            st.write("• JPG/JPEG")
            st.write("• PNG")
            st.write("• WebP")
        with col_b:
            st.write("**Videos:**")
            st.write("• MP4")
            st.write("• MOV")
            st.write("• AVI")
            st.write("• WebM")

# Footer
st.markdown("---")
st.caption(
    "💡 **Tips:** "
    "For faster video processing, increase the frame interval. "
    "Larger videos will take longer to analyze. "
    "The AI model may occasionally make mistakes - use results as guidance."
)

# Add system info in sidebar
with st.sidebar:
    st.header("ℹ️ System Information")
    st.write(f"**Model Input Size:** {IMG_WIDTH}x{IMG_HEIGHT}")
    st.write("**Framework:** TensorFlow/Keras")
    st.write("**Architecture:** MobileNetV2 + Custom Classifier")
    
    if model:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Not Loaded")
    
    st.header("📖 How to Use")
    st.write("1. Select content type (Image/Video)")
    st.write("2. Upload your file")
    st.write("3. Adjust settings if needed")
    st.write("4. View results and download report")