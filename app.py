import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image
import io
import imutils
from groq import Groq
import os

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 2rem;
    }
    .result-positive {
        background-color: #FFEBEE;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 1rem 0;
    }
    .result-negative {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class BrainTumorDetector:
    """Main class for brain tumor detection and processing"""
    
    def __init__(self):
        self.model = None
        self.groq_client = None
        
    @st.cache_resource
    def load_model(_self):
        """Load the pre-trained model"""
        try:
            from tensorflow.keras.models import load_model
            return load_model('brain_tumor_detector.h5')
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def init_groq(self, api_key):
        """Initialize Groq client"""
        try:
            self.groq_client = Groq(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Error initializing Groq: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for tumor detection"""
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)
        
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cnts) == 0:
            return None
            
        c = max(cnts, key=cv.contourArea)
        
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        new_image = img_array[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        
        if new_image.size == 0:
            return None
            
        resized = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
        normalized = resized / 255.0
        
        return normalized.reshape((1, 240, 240, 3))
    
    def predict_tumor(self, image):
        """Predict if tumor is present"""
        if self.model is None:
            self.model = self.load_model()
        
        if self.model is None:
            return None, "Model not loaded"
        
        preprocessed = self.preprocess_image(image)
        if preprocessed is None:
            return None, "Unable to preprocess image"
        
        prediction = self.model.predict(preprocessed, verbose=0)
        confidence = float(prediction[0][0])
        
        return confidence, "Success"
    
    def remove_noise(self, image):
        """Remove noise from image"""
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        
        return opening
    
    def segment_tumor(self, image):
        """Segment and display tumor region"""
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        # Noise removal
        opening = self.remove_noise(image)
        
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv.watershed(img_array, markers)
        img_array[markers == -1] = [255, 0, 0]
        
        return img_array
    
    def get_llm_analysis(self, has_tumor, confidence):
        """Get LLM analysis of the results"""
        if not self.groq_client:
            return "LLM analysis not available. Please provide a Groq API key."
        
        prompt = f"""You are a medical AI assistant specializing in brain tumor analysis. 
        
Based on the following brain MRI scan analysis results:
- Tumor Detected: {'Yes' if has_tumor else 'No'}
- Confidence Score: {confidence:.2%}

Please provide:
1. A brief interpretation of these results
2. What this confidence level means
3. Important next steps or recommendations
4. A reminder about the importance of professional medical consultation

Keep the response concise, professional, and empathetic. Use about 150-200 words."""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical AI assistant. Always remind users to consult healthcare professionals."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="openai/gpt-oss-120b",
                temperature=0.7,
                max_tokens=400
            )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating analysis: {str(e)}"


def main():
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = BrainTumorDetector()
    
    detector = st.session_state.detector
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Groq API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key for LLM analysis"
        )
        
        if groq_api_key:
            if detector.init_groq(groq_api_key):
                st.success("✅ Groq API connected")
        
        st.divider()
        
        st.header("📋 Analysis Options")
        analysis_mode = st.radio(
            "Select Analysis Type",
            ["Tumor Detection", "Tumor Segmentation", "Complete Analysis"],
            help="Choose what type of analysis to perform"
        )
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info(
            """
            This system uses deep learning to detect brain tumors in MRI scans.
            
            **Features:**
            - AI-powered tumor detection
            - Tumor region segmentation
            - LLM-powered result analysis
            
            ⚠️ **Important:** This is for educational purposes only. 
            Always consult medical professionals.
            """
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload MRI Scan</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original MRI Scan", use_container_width=True)
            
            # Store image in session state
            st.session_state.uploaded_image = image
    
    with col2:
        if 'uploaded_image' in st.session_state:
            st.markdown('<h2 class="sub-header">🔬 Analysis Results</h2>', unsafe_allow_html=True)
            
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing MRI scan..."):
                    image = st.session_state.uploaded_image
                    
                    if analysis_mode in ["Tumor Detection", "Complete Analysis"]:
                        # Tumor detection
                        confidence, status = detector.predict_tumor(image)
                        
                        if confidence is not None:
                            has_tumor = confidence > 0.5
                            
                            # Display result
                            if has_tumor:
                                st.markdown(
                                    f'<div class="result-positive">'
                                    f'<h3 style="color: #F44336; margin: 0;">⚠️ Tumor Detected</h3>'
                                    f'<p style="margin-top: 0.5rem; font-size: 1.1rem; color: black;">Confirmation: {confidence:.2%}</p>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div class="result-negative">'
                                    f'<h3 style="color: #4CAF50; margin: 0;">✅ No Tumor Detected</h3>'
                                    f'<p style="margin-top: 0.5rem; font-size: 1.1rem; color: black;">Confirmation: {(1-confidence):.2%}</p>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # LLM Analysis
                            if groq_api_key:
                                with st.expander("🤖 AI Analysis & Recommendations", expanded=True):
                                    analysis = detector.get_llm_analysis(has_tumor, confidence)
                                    st.write(analysis)
                            else:
                                st.warning("💡 Add Groq API key in sidebar for AI-powered analysis")
                            
                            # Store results for segmentation
                            st.session_state.detection_results = {
                                'has_tumor': has_tumor,
                                'confidence': confidence
                            }
                        else:
                            st.error(f"Analysis failed: {status}")
                    
                    if analysis_mode in ["Tumor Segmentation", "Complete Analysis"]:
                        if analysis_mode == "Tumor Segmentation" or (
                            'detection_results' in st.session_state and 
                            st.session_state.detection_results['has_tumor']
                        ):
                            st.divider()
                            st.subheader("🎯 Tumor Segmentation")
                            
                            # Remove noise visualization
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.write("**Noise Removal**")
                                denoised = detector.remove_noise(image)
                                st.image(denoised, caption="After Noise Removal", use_container_width=True)
                            
                            with col_b:
                                st.write("**Tumor Region**")
                                segmented = detector.segment_tumor(image)
                                st.image(segmented, caption="Segmented Tumor (Red)", use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>Disclaimer:</strong> This tool is for educational and research purposes only. 
            It should not be used as a substitute for professional medical diagnosis or treatment.</p>
            <p>Always consult qualified healthcare professionals for medical advice.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()