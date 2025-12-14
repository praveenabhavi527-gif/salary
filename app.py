import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Professional Salary Predictor",
    page_icon="Ws",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DESIGN ---
# This injects CSS to style the app (Gradient background, card styles, custom buttons)
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Card Container Style */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    /* Header Styling */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    h3 {
        color: #34495e;
        font-weight: 400;
    }

    /* Custom Button Styling */
    div.stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 18px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Result Box Styling */
    .result-box {
        background-color: #2c3e50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Input Field Styling */
    .stNumberInput input {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("About")
    st.info(
        """
        This AI model utilizes **Linear Regression** to predict expected salary based on years of professional experience.
        
        **Model Info:**
        - Library: Scikit-learn
        - Data Source: Salary Data.csv
        """
    )
    st.write("---")
    st.caption("© 2024 Salary AI Inc.")

# --- MAIN APP LOGIC ---

def load_model():
    """Loads the model, handling errors if the file is missing."""
    try:
        # Check current directory
        if os.path.exists('salary_model.pkl'):
            with open('salary_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            st.error("Model file 'salary_model.pkl' not found in the directory.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.markdown("<h1>💰 Salary Prediction AI</h1>", unsafe_allow_html=True)
    st.write("### estimate your market value instantly")
    
    # Create a layout with a central card look
    with st.container():
        st.write("---")
        
        # Input Section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            years_exp = st.number_input(
                "Enter Years of Experience:",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="Enter the total number of professional working years."
            )

        # Space
        st.write("")
        st.write("")

        # Predict Button
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            predict_btn = st.button("Calculate Salary")

        # Prediction Logic
        if predict_btn:
            model = load_model()
            
            if model:
                # Making prediction
                # Reshape input because sklearn expects a 2D array [[value]]
                input_data = np.array([[years_exp]])
                
                try:
                    prediction = model.predict(input_data)
                    result = round(prediction[0], 2)
                    
                    # Display Result with custom CSS class
                    st.markdown(f"""
                        <div class="result-box">
                            <div>Estimated Annual Salary</div>
                            <div style="font-size: 40px; font-weight: bold;">${result:,.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.warning("⚠️ The model expects valid feature names or the shape of the data doesn't match.")
                    st.error(f"Detailed Error: {e}")

if __name__ == '__main__':
    main()
