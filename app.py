import streamlit as st
import joblib
import numpy as np
import time
import base64
from pathlib import Path

# Function to load and encode image to Base64
def img_to_base64(img_path):
    """Converts an image file to a Base64 encoded string."""
    try:
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except FileNotFoundError:
        st.error(f"Image file not found at {img_path}")
        return ""
    
# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
)

# ---------------- Load Model ----------------
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

try:
    with open("accuracy.txt", "r") as f:
        train_acc, test_acc = f.read().split(",")
        train_acc, test_acc = float(train_acc), float(test_acc)
except FileNotFoundError:
    train_acc = test_acc = None

# ---------------- Sidebar Navigation ----------------
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Go to", ["About", "News Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption("📰 Fake News Detector AI")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
/* App background with subtle gradient and texture */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #0f172a, #222e40);  #,#1e293b
    color: white;
    background-image: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.05) 0%, transparent 20%),
                      radial-gradient(circle at 80% 80%, rgba(255,255,255,0.05) 0%, transparent 20%);
}

            /* Hero section */
.hero {
    background: linear-gradient(90deg, #6d6fe9, #2668d0);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
    box-shadow: 0px 5px 25px rgba(99,102,241,0.3);
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 3em;
    font-weight: 800;
    margin: 0;
}
.hero p {
    font-size: 1.2em;
    opacity: 0.9;
}

/* Text area styling */
.stTextArea textarea {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 12px;
    color: #e2e8f0;
    font-size: 1rem;
    padding: 1rem;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.stTextArea textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.5);
}

/* Button styling */
.stButton button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.8rem;
    transition: all 0.3s ease;
}
.stButton button:hover {
    background: linear-gradient(90deg, #818cf8, #60a5fa);
    transform: scale(1.05);
}
            
/* Result card styling */
.result-card {
    border-radius: 14px;
    padding: 1.2rem;
    margin-top: 1rem;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}
.fake { background: rgba(239,68,68,0.15); border-left: 5px solid #ef4444; }
.real { background: rgba(34,197,94,0.15); border-left: 5px solid #22c55e; }
.metric-card {
    padding: 1rem;
    border-radius: 14px;
    text-align: center;
    color: #fff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
.train { background: linear-gradient(135deg, #14b8a6, #0d9488); }
.test { background: linear-gradient(135deg, #3b82f6, #2563eb); }
</style>
""", unsafe_allow_html=True)

# ---------------- About Page ----------------
if page == "About":
    st.markdown("""
        <style>
        /*Layout */
    
        h1, h2, h3 {
            color: #f8fafc;
            font-family: 'Poppins', sans-serif;
        }

        .hero-section {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 2rem;
            margin-bottom: 2rem;
            gap: 2rem; /* --- EDIT: Added gap for spacing --- */
        }

        .hero-text {
            flex: 1;
            padding-right: 2rem;
        }

        .hero-image {
            flex: 1;
            text-align: center;
        }

        .hero-image img {
            width: 80%;
            border-radius: 15px; /* --- EDIT: Added for softer corners --- */
            filter: drop-shadow(0 0 20px rgba(59,130,246,0.4));
        }

        .process-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 2rem;
            margin-bottom: 2rem;
            gap: 1.5rem; /* --- EDIT: Added gap for spacing --- */
        }

        .process-box {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 1.5rem; /* --- EDIT: Increased padding --- */
            text-align: center;
            width: 30%;
            box-shadow: 0px 0px 10px rgba(59,130,246,0.4);
            transition: all 0.3s ease;
            
            /* --- EDIT START: Added shiny top border effect --- */
            border-top: 3px solid transparent;
            border-image: linear-gradient(to right, #3b82f6, rgba(59,130,246,0)) 1;
            border-image-slice: 1;
            /* --- EDIT END --- */
        }

        .process-box:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 15px rgba(147,197,253,0.7);
        }

        .process-box h4 {
            color: #93c5fd;
            margin-bottom: 0.5rem;
        }

        .overview-list {
            list-style: none;
            padding-left: 0;
            font-size: 1.1rem;
            line-height: 1.8;
        }

        .overview-list li::before {
            content: "⚙️ ";
        }

        .info-box {
            background: rgba(16, 185, 129, 0.15);
            border-left: 5px solid #10b981;
            padding: 1rem;
            border-radius: 10px;
            color: #d1fae5;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Encode the image to Base64 to embed it in HTML
    img_base64 = img_to_base64("AI News Icon.jpg")
    

    # Header
    # Modified to be an f-string and added the hero-image div ---
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-text">
            <h1>🛡️ Fake News Detector AI</h1>
            <h3>AI-Powered News Authenticity Analysis</h3>
            <p>
                Fake News Detector AI uses advanced NLP and machine learning to assess
                the authenticity of online news articles. It helps journalists, researchers, 
                and readers identify misinformation quickly and effectively.
            </p>
        </div>
        <div class="hero-image">
            <img src="data:image/jpeg;base64,{img_base64}" alt="AI News Icon">
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)
    
    # Overview section
    st.markdown("""
    <h3>⚙️ Overview</h3>
    <ul class="overview-list">
        <li>Python 🐍</li>
        <li>Streamlit 🌐</li>
        <li>Scikit-learn ⚙️</li>
    </ul>
    """, unsafe_allow_html=True)

    # Process flow
    st.markdown("""
    <div class="process-container">
        <div class="process-box">
            <h4>🧹 Preprocessing</h4>
            <p>Text is cleaned, tokenized, and vectorized to prepare for analysis.</p>
        </div>
        <div class="process-box">
            <h4>🤖 Prediction</h4>
            <p>Logistic Regression model predicts whether the article is real or fake.</p>
        </div>
        <div class="process-box">
            <h4>📊 Confidence</h4>
            <p>The model outputs a confidence score showing how certain it is.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Info message
    st.markdown("""
    <div class="info-box">
        👈🏻 <b>Select 'News Prediction'</b> from the sidebar to begin testing the model.
    </div>
    """, unsafe_allow_html=True)


    
# ---------------- News Prediction Page ----------------
elif page == "News Prediction":
    # -------------------- Hero Header --------------------
    st.markdown("""
    <div class="hero">
        <h1>📰 Fake News Detector</h1>
        <p>Detect misinformation instantly using AI — analyze the trustworthiness of any news article.</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------- Input Section --------------------
    news_input = st.text_area("✏️ Paste or type a news article:", height=180)

    # -------------------- Check News Button --------------------
    if st.button("Check News"):
        if news_input.strip():
            with st.spinner("Analyzing article... 🔎"):
                time.sleep(1.5)
                transform_input = vectorizer.transform([news_input])
                prediction = model.predict(transform_input)[0]
                proba = model.predict_proba(transform_input)[0]
                confidence = np.max(proba) * 100

            if prediction == 1:
                st.markdown(f"""
                <div class='result-card real'>
                    ✅ <b>The article appears REAL</b><br>
                    <i>Confidence:</i> {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-card fake'>
                    🚨 <b>The article appears FAKE</b><br>
                    <i>Confidence:</i> {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)
            
            # ---------------- Model Dashboard Inside ----------------
            st.markdown("### 📊 Model Performance Overview")
            st.write("Compare how the model performed on training vs unseen testing data.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card train">
                    🧠 <b>Training Accuracy</b>
                    <h2>{train_acc:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card test">
                    📈 <b>Testing Accuracy</b>
                    <h2>{test_acc:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("⚠️ Please enter some text to analyze.")
