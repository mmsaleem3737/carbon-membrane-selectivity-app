import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ Page configuration - must be first
st.set_page_config(
    page_title="CO₂/N₂ Selectivity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Define the custom transformer BEFORE loading the model
class RarePolymerGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=5):
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
        else:
            col = pd.Series(X[:, 0])
        counts = col.value_counts()
        self.frequent_polymers_ = set(counts[counts >= self.threshold].index)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
        else:
            col = pd.Series(X[:, 0])
        grouped = col.apply(lambda x: x if x in self.frequent_polymers_ else "Other")
        return grouped.to_frame()

# ✅ Load the pipeline using joblib — class now known to joblib
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("PolyMemCO2Pipeline.joblib")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'PolyMemCO2Pipeline.joblib' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# 🎯 Create gauge chart for visualization
def create_gauge_chart(value, title="CO₂/N₂ Selectivity"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#2c3e50'}},
        delta = {'reference': 50, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 200], 'tickcolor': "#2c3e50"},
            'bar': {'color': "#1e3c72"},
            'steps': [
                {'range': [0, 25], 'color': "#ffcccb"},
                {'range': [25, 50], 'color': "#fff2cc"},
                {'range': [50, 100], 'color': "#d4edda"},
                {'range': [100, 200], 'color': "#d1ecf1"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "#2c3e50"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# 🔃 Load the model into a global variable
model = load_pipeline()

# 🏠 Main App Header
st.title("🧪 CO₂/N₂ Selectivity Predictor")
st.markdown("### Advanced Machine Learning Model for Polymer Membrane Analysis")

# 📋 Sidebar with information
with st.sidebar:
    st.markdown("## 📖 About This App")
    st.markdown("""
    This application predicts the **CO₂/N₂ selectivity** of polymer membranes using advanced machine learning techniques.
    
    ### 🎯 How it works:
    1. **Input Parameters**: Enter polymer properties and gas permeability values
    2. **ML Prediction**: Our trained model analyzes your inputs
    3. **Results**: Get instant selectivity predictions with confidence metrics
    
    ### 📊 Input Guidelines:
    - **Permeability Units**: Barrer (10⁻¹⁰ cm³(STP)·cm/(cm²·s·cmHg))
    - **Typical Ranges**:
        - He: 1-100 Barrer
        - H₂: 1-50 Barrer
        - CO₂: 1-200 Barrer
        - O₂: 0.1-20 Barrer
        - N₂: 0.01-5 Barrer
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Model Information")
    st.info("This model was trained on a comprehensive dataset of polymer membrane properties with 1,407 samples and 63 features, achieving an R² of 0.9175.")

# Initialize session state for input values
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.inputs = {
        'He_Permeability': 0.0,
        'H2_Permeability': 0.0,
        'O2_Permeability': 0.0,
        'N2_Permeability': 0.0,
        'CO2_Permeability': 0.0,
        'CH4_Permeability': 0.0,
        'C2H4_Permeability': 0.0,
        'C2H6_Permeability': 0.0,
        'C3H6_Permeability': 0.0,
        'C3H8_Permeability': 0.0,
        'C4H8_Permeability': 0.0,
        'nC4H10_Permeability': 0.0,
        'CF4_Permeability': 0.0,
        'C2F6_Permeability': 0.0,
        'C3F8_Permeability': 0.0
    }

# 📥 Main input section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧬 Polymer Configuration")
    
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        polymer_type_options = [
            "Polyimides and Polypyrrolones", "Polymers with High Free Volume", 
            "Thermally Rearranged Polymers", "Silicone rubber and variants",
            "Substituted Polyacetylenes", "Mixed Matrix Membranes", "Other"
        ]
        polymer_type_choice = st.selectbox("**Polymer Type**", polymer_type_options, index=0, help="Select the category of polymer membrane")
        if polymer_type_choice == "Other":
            polymer_type = st.text_input("Custom Polymer Type:", placeholder="Enter your polymer type...", help="Specify your custom polymer type")
        else:
            polymer_type = polymer_type_choice
    with subcol2:
        polymer_options = [
            "Polyimide", "Polysulfone", "Polyetherimide", "Poly(vinyl chloride)", 
            "Polytrimethylsilylpropyne", "Polypyrrolone", "Cellulose acetate", 
            "Polycarbonate", "Polyethersulfone", "Other"
        ]
        polymer_choice = st.selectbox("**Specific Polymer**", polymer_options, index=0, help="Select the specific polymer material")
        if polymer_choice == "Other":
            polymer = st.text_input("Custom Polymer:", placeholder="Enter your polymer name...", help="Specify your custom polymer")
        else:
            polymer = polymer_choice

    st.subheader("⚛️ Gas Permeability Values")
    
    with st.expander("🌟 Primary Gases", expanded=True):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            He_Permeability = st.number_input("He Permeability", min_value=0.0, value=st.session_state.inputs['He_Permeability'], format="%.3f", key="he")
            H2_Permeability = st.number_input("H₂ Permeability", min_value=0.0, value=st.session_state.inputs['H2_Permeability'], format="%.3f", key="h2")
        with pcol2:
            O2_Permeability = st.number_input("O₂ Permeability", min_value=0.0, value=st.session_state.inputs['O2_Permeability'], format="%.3f", key="o2")
            N2_Permeability = st.number_input("N₂ Permeability", min_value=0.0, value=st.session_state.inputs['N2_Permeability'], format="%.3f", key="n2")
        with pcol3:
            CO2_Permeability = st.number_input("CO₂ Permeability", min_value=0.0, value=st.session_state.inputs['CO2_Permeability'], format="%.3f", key="co2")
    
    with st.expander("🔬 Hydrocarbons", expanded=False):
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            CH4_Permeability = st.number_input("CH₄ Permeability", min_value=0.0, value=st.session_state.inputs['CH4_Permeability'], format="%.3f", key="ch4")
            C2H4_Permeability = st.number_input("C₂H₄ Permeability", min_value=0.0, value=st.session_state.inputs['C2H4_Permeability'], format="%.3f", key="c2h4")
            C2H6_Permeability = st.number_input("C₂H₆ Permeability", min_value=0.0, value=st.session_state.inputs['C2H6_Permeability'], format="%.3f", key="c2h6")
            C3H6_Permeability = st.number_input("C₃H₆ Permeability", min_value=0.0, value=st.session_state.inputs['C3H6_Permeability'], format="%.3f", key="c3h6")
        with hcol2:
            C3H8_Permeability = st.number_input("C₃H₈ Permeability", min_value=0.0, value=st.session_state.inputs['C3H8_Permeability'], format="%.3f", key="c3h8")
            C4H8_Permeability = st.number_input("C₄H₈ Permeability", min_value=0.0, value=st.session_state.inputs['C4H8_Permeability'], format="%.3f", key="c4h8")
            nC4H10_Permeability = st.number_input("n-C₄H₁₀ Permeability", min_value=0.0, value=st.session_state.inputs['nC4H10_Permeability'], format="%.3f", key="nc4h10")
    
    with st.expander("🧪 Fluorocarbons", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            CF4_Permeability = st.number_input("CF₄ Permeability", min_value=0.0, value=st.session_state.inputs['CF4_Permeability'], format="%.3f", key="cf4")
        with fcol2:
            C2F6_Permeability = st.number_input("C₂F₆ Permeability", min_value=0.0, value=st.session_state.inputs['C2F6_Permeability'], format="%.3f", key="c2f6")
        with fcol3:
            C3F8_Permeability = st.number_input("C₃F₈ Permeability", min_value=0.0, value=st.session_state.inputs['C3F8_Permeability'], format="%.3f", key="c3f8")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 **Predict CO₂/N₂ Selectivity**", use_container_width=True)

with col2:
    st.subheader("📊 Quick Stats & Results")
    
    # Model performance metrics
    st.metric("Model R²", "0.9175", "High accuracy")
    
    if CO2_Permeability > 0 and N2_Permeability > 0:
        experimental_ratio = CO2_Permeability / N2_Permeability
        st.metric("Experimental CO₂/N₂", f"{experimental_ratio:.2f}", "Based on your inputs")

    # 🔮 Prediction logic
    if predict_button and model is not None:
        validation_errors = []
        if polymer_type_choice == "Other" and not polymer_type:
            validation_errors.append("Please enter a custom Polymer Type.")
        if polymer_choice == "Other" and not polymer:
            validation_errors.append("Please enter a custom Polymer.")
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
        else:
            with st.spinner('🔄 Analyzing...'):
                time.sleep(1.5)
                try:
                    input_data = {
                        "Polymer Type": [polymer_type], "Polymer": [polymer],
                        "He_Permeability": [He_Permeability], "H2_Permeability": [H2_Permeability],
                        "O2_Permeability": [O2_Permeability], "N2_Permeability": [N2_Permeability],
                        "CO2_Permeability": [CO2_Permeability], "CH4_Permeability": [CH4_Permeability],
                        "C2H4_Permeability": [C2H4_Permeability], "C2H6_Permeability": [C2H6_Permeability],
                        "C3H6_Permeability": [C3H6_Permeability], "C3H8_Permeability": [C3H8_Permeability],
                        "C4H8_Permeability": [C4H8_Permeability], "n-C4H10_Permeability": [nC4H10_Permeability],
                        "CF4_Permeability": [CF4_Permeability], "C2F6_Permeability": [C2F6_Permeability],
                        "C3F8_Permeability": [C3F8_Permeability]
                    }
                    input_df = pd.DataFrame(input_data)
                    prediction = model.predict(input_df)[0]
                    
                    st.success(f"🎯 **Predicted CO₂/N₂ Selectivity: {prediction:.2f}**")
                    
                    gauge_fig = create_gauge_chart(prediction)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    if prediction < 10:
                        interpretation = "⚠️ **Low Selectivity** - May not be suitable for efficient separation."
                    elif prediction < 30:
                        interpretation = "📊 **Moderate Selectivity** - Shows reasonable separation performance."
                    elif prediction < 50:
                        interpretation = "✅ **Good Selectivity** - Demonstrates good separation efficiency."
                    else:
                        interpretation = "🌟 **Excellent Selectivity** - Shows outstanding separation performance."
                    st.info(interpretation)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

    elif predict_button and model is None:
        st.error("❌ Model not loaded. Check file path.")

# 📚 Additional information sections at the bottom
st.markdown("---")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    with st.expander("🔬 About CO₂/N₂ Selectivity"):
        st.markdown("""
        **CO₂/N₂ Selectivity** is a crucial parameter in gas separation membranes, representing the ability of a membrane to preferentially allow CO₂ to pass through while blocking N₂.
        - Higher selectivity = Better separation efficiency
        - Typical industrial targets: 20-50+
        """)

with col_info2:
    with st.expander("📊 Model Performance Details"):
        st.markdown("""
        Our machine learning model was trained on a comprehensive dataset of polymer membrane properties.
        - **Training samples:** 1,407
        - **Features:** 63 input parameters
        - **MAE:** 1.76
        - **RMSE:** 3.11
        """)

with col_info3:
    with st.expander("🔧 App & Model Info"):
        st.markdown("""
        This app uses a Scikit-learn pipeline and a RandomForestRegressor model.
        - **Developer:** Saleem
        - **Version:** 1.1.0
        """)

# Footer
st.markdown("---")
st.markdown("*Developed with ❤️ using Streamlit & Scikit-learn | For educational and research purposes only.*")