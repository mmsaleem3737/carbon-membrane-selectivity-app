import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
import os
import logging
from typing import Optional, Tuple, Dict, Any

# ✅ Page configuration - must be first
st.set_page_config(
    page_title="CO₂/N₂ Selectivity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Define the custom transformer BEFORE loading the model
class RarePolymerGrouper(BaseEstimator, TransformerMixin):
    """Custom transformer for grouping rare polymers."""
    
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.frequent_polymers_ = set()

    def fit(self, X, y=None):
        """Fit the transformer by identifying frequent polymers."""
        try:
            if isinstance(X, pd.DataFrame):
                col = X.iloc[:, 0]
            else:
                col = pd.Series(X[:, 0]) if len(X.shape) > 1 else pd.Series(X)
            
            counts = col.value_counts()
            self.frequent_polymers_ = set(counts[counts >= self.threshold].index)
            logger.info(f"Identified {len(self.frequent_polymers_)} frequent polymers")
            return self
        except Exception as e:
            logger.error(f"Error in RarePolymerGrouper.fit: {e}")
            self.frequent_polymers_ = set()
            return self

    def transform(self, X):
        """Transform data by grouping rare polymers."""
        try:
            if isinstance(X, pd.DataFrame):
                col = X.iloc[:, 0]
            else:
                col = pd.Series(X[:, 0]) if len(X.shape) > 1 else pd.Series(X)
            
            grouped = col.apply(lambda x: x if x in self.frequent_polymers_ else "Other")
            return grouped.to_frame() if isinstance(X, pd.DataFrame) else grouped.values.reshape(-1, 1)
        except Exception as e:
            logger.error(f"Error in RarePolymerGrouper.transform: {e}")
            return X

# ✅ Enhanced model loading with comprehensive error handling
@st.cache_resource
def load_pipeline() -> Tuple[Optional[Any], Optional[str]]:
    """Load the ML pipeline with multiple fallback methods."""
    
    model_files = [
        "PolyMemCO2Pipeline.joblib", 
        "PolyMemCO2Pipeline.pkl", 
        "model.joblib", 
        "model.pkl",
        "pipeline.joblib",
        "pipeline.pkl"
    ]
    
    # Check what files actually exist
    existing_files = [f for f in model_files if os.path.exists(f)]
    all_files = [f for f in os.listdir('.') if f.endswith(('.joblib', '.pkl'))]
    
    logger.info(f"Looking for model files: {model_files}")
    logger.info(f"Found existing model files: {existing_files}")
    logger.info(f"All .joblib/.pkl files in directory: {all_files}")
    
    # Try loading existing files first
    for model_file in existing_files:
        # Method 1: Try joblib first
        try:
            model = joblib.load(model_file)
            logger.info(f"Successfully loaded model with joblib from {model_file}")
            
            # Validate model has predict method
            if hasattr(model, 'predict'):
                st.success(f"✅ Model loaded successfully with joblib from {model_file}!")
                return model, model_file
            else:
                logger.warning(f"Loaded object from {model_file} doesn't have predict method")
                continue
                
        except Exception as e1:
            logger.warning(f"joblib failed for {model_file}: {str(e1)[:150]}")
            
            # Method 2: Try pickle as fallback
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Successfully loaded model with pickle from {model_file}")
                
                # Validate model has predict method
                if hasattr(model, 'predict'):
                    st.success(f"✅ Model loaded successfully with pickle from {model_file}!")
                    return model, model_file
                else:
                    logger.warning(f"Loaded object from {model_file} doesn't have predict method")
                    continue
                    
            except Exception as e2:
                logger.error(f"Both joblib and pickle failed for {model_file}")
                logger.error(f"joblib error: {str(e1)[:100]}")
                logger.error(f"pickle error: {str(e2)[:100]}")
                continue
    
    # If no model file found, show comprehensive error
    st.error("❌ No compatible model file found!")
    
    with st.expander("🔍 Detailed Diagnostics", expanded=True):
        st.error(f"**Searched for files:** {', '.join(model_files)}")
        
        if all_files:
            st.warning(f"**Found .joblib/.pkl files:** {', '.join(all_files)}")
            st.info("💡 Try renaming one of the found files to match expected names")
        else:
            st.error("**No .joblib or .pkl files found in current directory**")
        
        # Show sklearn version info
        try:
            import sklearn
            st.info(f"**Current sklearn version:** {sklearn.__version__}")
            st.markdown("""
            **Possible solutions:**
            1. Ensure model file exists and is named correctly
            2. Check sklearn version compatibility
            3. Retrain model with current sklearn version
            4. Verify model was saved properly
            """)
        except ImportError:
            st.error("**sklearn not available** - Please install scikit-learn")
    
    return None, None

# 🎯 Enhanced gauge chart with better styling
def create_gauge_chart(value: float, title: str = "CO₂/N₂ Selectivity") -> go.Figure:
    """Create an enhanced gauge chart for selectivity visualization."""
    
    # Ensure value is numeric and handle edge cases
    value = float(value) if value is not None and not np.isnan(value) else 0
    value = max(0, min(300, value))  # Clamp between 0 and 300 for better visualization
    
    # Define performance ranges
    ranges = [
        {"range": [0, 10], "color": "#ff6b6b", "label": "Poor"},
        {"range": [10, 25], "color": "#feca57", "label": "Fair"},
        {"range": [25, 50], "color": "#48dbfb", "label": "Good"},
        {"range": [50, 100], "color": "#0be881", "label": "Excellent"},
        {"range": [100, 300], "color": "#0066cc", "label": "Outstanding"}
    ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>{title}</b>", 
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        delta={
            'reference': 30, 
            'position': "top",
            'font': {'size': 16}
        },
        number={'font': {'size': 32, 'color': '#2c3e50'}},
        gauge={
            'axis': {
                'range': [None, 300], 
                'tickcolor': "#2c3e50",
                'tickfont': {'size': 14}
            },
            'bar': {'color': "#1e3c72", 'thickness': 0.3},
            'steps': [
                {'range': r["range"], 'color': r["color"]} 
                for r in ranges
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': 100
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "#2c3e50", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# 📊 Create comparison chart
def create_comparison_chart(experimental: float, predicted: float) -> go.Figure:
    """Create a comparison chart between experimental and predicted values."""
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Experimental\n(CO₂/N₂)', 'Predicted\n(ML Model)'],
            y=[experimental, predicted],
            marker_color=['#3498db', '#e74c3c'],
            text=[f'{experimental:.2f}', f'{predicted:.2f}'],
            textposition='auto',
            textfont=dict(size=16, color='white')
        )
    ])
    
    fig.update_layout(
        title="<b>Experimental vs Predicted Selectivity</b>",
        title_font_size=18,
        yaxis_title="CO₂/N₂ Selectivity",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# 🔄 Initialize session state
def initialize_session_state():
    """Initialize session state with default values."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.inputs = {
            'He_Permeability': 10.0,
            'H2_Permeability': 8.0,
            'O2_Permeability': 2.0,
            'N2_Permeability': 0.5,
            'CO2_Permeability': 15.0,
            'CH4_Permeability': 0.3,
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
        st.session_state.prediction_made = False
        st.session_state.last_prediction = None

# 🎨 Custom CSS for better styling
def load_custom_css():
    """Load custom CSS for enhanced styling."""
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1em;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# 🏠 Main application
def main():
    """Main application function."""
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Load model
    model, model_file = load_pipeline()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧪 CO₂/N₂ Selectivity Predictor</h1>
        <p>Advanced Machine Learning Model for Polymer Membrane Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sklearn version and model info in sidebar
    with st.sidebar:
        st.markdown("## 🔧 System Information")
        
        try:
            import sklearn
            st.success(f"📊 scikit-learn: {sklearn.__version__}")
            st.session_state['sklearn_version'] = sklearn.__version__
        except ImportError:
            st.error("❌ scikit-learn not available")
        
        if model_file:
            st.success(f"📁 Model: {model_file}")
            
        # Model performance info
        st.markdown("## 📈 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", "0.9175")
            st.metric("MAE", "1.76")
        with col2:
            st.metric("RMSE", "3.11")
            st.metric("Samples", "1,407")
        
        st.markdown("---")
        st.markdown("## 📖 About This App")
        st.markdown("""
        This application predicts **CO₂/N₂ selectivity** of polymer membranes using advanced machine learning.
        
        ### 🎯 Key Features:
        - Real-time ML predictions
        - Interactive visualizations
        - Comprehensive input validation
        - Performance comparison tools
        
        ### 📊 Input Guidelines:
        **Permeability Units**: Barrer (10⁻¹⁰ cm³(STP)·cm/(cm²·s·cmHg))
        
        **Typical Ranges**:
        - He: 1-100 Barrer
        - H₂: 1-50 Barrer  
        - CO₂: 1-200 Barrer
        - O₂: 0.1-20 Barrer
        - N₂: 0.01-5 Barrer
        """)

    # Only show main app if model loaded successfully
    if model is not None:
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Polymer configuration section
            st.markdown("### 🧬 Polymer Configuration")
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                polymer_type_options = [
                    "Polyimides and Polypyrrolones", 
                    "Polymers with High Free Volume", 
                    "Thermally Rearranged Polymers", 
                    "Silicone rubber and variants",
                    "Substituted Polyacetylenes", 
                    "Mixed Matrix Membranes", 
                    "Other"
                ]
                polymer_type_choice = st.selectbox(
                    "**Polymer Type**", 
                    polymer_type_options, 
                    index=0, 
                    help="Select the category of polymer membrane"
                )
                
                if polymer_type_choice == "Other":
                    polymer_type = st.text_input(
                        "Custom Polymer Type:", 
                        placeholder="Enter your polymer type...", 
                        help="Specify your custom polymer type"
                    )
                else:
                    polymer_type = polymer_type_choice
                    
            with subcol2:
                polymer_options = [
                    "Polyimide", "Polysulfone", "Polyetherimide", 
                    "Poly(vinyl chloride)", "Polytrimethylsilylpropyne", 
                    "Polypyrrolone", "Cellulose acetate", "Polycarbonate", 
                    "Polyethersulfone", "Other"
                ]
                polymer_choice = st.selectbox(
                    "**Specific Polymer**", 
                    polymer_options, 
                    index=0, 
                    help="Select the specific polymer material"
                )
                
                if polymer_choice == "Other":
                    polymer = st.text_input(
                        "Custom Polymer:", 
                        placeholder="Enter your polymer name...", 
                        help="Specify your custom polymer"
                    )
                else:
                    polymer = polymer_choice

            # Gas permeability inputs
            st.markdown("### ⚛️ Gas Permeability Values")
            
            # Primary gases (always expanded)
            with st.expander("🌟 Primary Gases", expanded=True):
                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    He_Permeability = st.number_input(
                        "He Permeability", 
                        min_value=0.0, 
                        value=st.session_state.inputs['He_Permeability'], 
                        format="%.3f", 
                        key="he",
                        help="Helium permeability in Barrer"
                    )
                    H2_Permeability = st.number_input(
                        "H₂ Permeability", 
                        min_value=0.0, 
                        value=st.session_state.inputs['H2_Permeability'], 
                        format="%.3f", 
                        key="h2",
                        help="Hydrogen permeability in Barrer"
                    )
                with pcol2:
                    O2_Permeability = st.number_input(
                        "O₂ Permeability", 
                        min_value=0.0, 
                        value=st.session_state.inputs['O2_Permeability'], 
                        format="%.3f", 
                        key="o2",
                        help="Oxygen permeability in Barrer"
                    )
                    N2_Permeability = st.number_input(
                        "N₂ Permeability", 
                        min_value=0.0, 
                        value=st.session_state.inputs['N2_Permeability'], 
                        format="%.3f", 
                        key="n2",
                        help="Nitrogen permeability in Barrer"
                    )
                with pcol3:
                    CO2_Permeability = st.number_input(
                        "CO₂ Permeability", 
                        min_value=0.0, 
                        value=st.session_state.inputs['CO2_Permeability'], 
                        format="%.3f", 
                        key="co2",
                        help="Carbon dioxide permeability in Barrer"
                    )
            
            # Hydrocarbons (collapsible)
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
            
            # Fluorocarbons (collapsible)
            with st.expander("🧪 Fluorocarbons", expanded=False):
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    CF4_Permeability = st.number_input("CF₄ Permeability", min_value=0.0, value=st.session_state.inputs['CF4_Permeability'], format="%.3f", key="cf4")
                with fcol2:
                    C2F6_Permeability = st.number_input("C₂F₆ Permeability", min_value=0.0, value=st.session_state.inputs['C2F6_Permeability'], format="%.3f", key="c2f6")
                with fcol3:
                    C3F8_Permeability = st.number_input("C₃F₈ Permeability", min_value=0.0, value=st.session_state.inputs['C3F8_Permeability'], format="%.3f", key="c3f8")

            # Prediction button
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Button layout
            bcol1, bcol2, bcol3 = st.columns([1, 2, 1])
            with bcol2:
                predict_button = st.button(
                    "🔮 **Predict CO₂/N₂ Selectivity**", 
                    use_container_width=True,
                    type="primary"
                )

        with col2:
            st.markdown("### 📊 Results & Analysis")
            
            # Quick metrics
            if CO2_Permeability > 0 and N2_Permeability > 0:
                experimental_ratio = CO2_Permeability / N2_Permeability
                st.markdown(f"""
                <div class="metric-container">
                    <h4>🧮 Experimental CO₂/N₂</h4>
                    <h2 style="color: #007bff; margin: 0;">{experimental_ratio:.2f}</h2>
                    <small>Based on your input values</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Show prediction results if available
            if st.session_state.get('prediction_made', False) and st.session_state.get('last_prediction') is not None:
                prediction = st.session_state.last_prediction
                
                st.markdown(f"""
                <div class="success-box">
                    🎯 ML Predicted: {prediction:.2f}
                </div>
                """, unsafe_allow_html=True)
                
                # Create and display gauge chart
                gauge_fig = create_gauge_chart(prediction)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Show comparison if experimental value exists
                if CO2_Permeability > 0 and N2_Permeability > 0:
                    comparison_fig = create_comparison_chart(experimental_ratio, prediction)
                    st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Interpretation
                if prediction < 10:
                    interpretation = "⚠️ **Low Selectivity** - May not be suitable for efficient separation."
                    color = "#ff6b6b"
                elif prediction < 25:
                    interpretation = "📊 **Fair Selectivity** - Shows reasonable separation performance."
                    color = "#feca57"
                elif prediction < 50:
                    interpretation = "✅ **Good Selectivity** - Demonstrates good separation efficiency."
                    color = "#48dbfb"
                elif prediction < 100:
                    interpretation = "🌟 **Excellent Selectivity** - Shows outstanding separation performance."
                    color = "#0be881"
                else:
                    interpretation = "🚀 **Outstanding Selectivity** - Exceptional separation performance!"
                    color = "#0066cc"
                
                st.markdown(f"""
                <div style="background: {color}; color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;">
                    {interpretation}
                </div>
                """, unsafe_allow_html=True)

        # Prediction logic
        if predict_button:
            # Validate inputs
            validation_errors = []
            if polymer_type_choice == "Other" and not polymer_type.strip():
                validation_errors.append("Please enter a custom Polymer Type.")
            if polymer_choice == "Other" and not polymer.strip():
                validation_errors.append("Please enter a custom Polymer.")
            
            if validation_errors:
                for error in validation_errors:
                    st.error(f"❌ {error}")
            else:
                with st.spinner('🔄 Analyzing polymer membrane properties...'):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    try:
                        # Prepare input data
                        input_data = {
                            "Polymer Type": [polymer_type], 
                            "Polymer": [polymer],
                            "He_Permeability": [He_Permeability], 
                            "H2_Permeability": [H2_Permeability],
                            "O2_Permeability": [O2_Permeability], 
                            "N2_Permeability": [N2_Permeability],
                            "CO2_Permeability": [CO2_Permeability], 
                            "CH4_Permeability": [CH4_Permeability],
                            "C2H4_Permeability": [C2H4_Permeability], 
                            "C2H6_Permeability": [C2H6_Permeability],
                            "C3H6_Permeability": [C3H6_Permeability], 
                            "C3H8_Permeability": [C3H8_Permeability],
                            "C4H8_Permeability": [C4H8_Permeability], 
                            "n-C4H10_Permeability": [nC4H10_Permeability],
                            "CF4_Permeability": [CF4_Permeability], 
                            "C2F6_Permeability": [C2F6_Permeability],
                            "C3F8_Permeability": [C3F8_Permeability]
                        }
                        input_df = pd.DataFrame(input_data)
                        
                        # Make prediction
                        prediction = model.predict(input_df)[0]
                        prediction = float(prediction)
                        
                        # Store prediction in session state
                        st.session_state.prediction_made = True
                        st.session_state.last_prediction = prediction
                        
                        st.success(f"✅ Prediction completed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        logger.error(f"Prediction error: {e}")
                        st.error("Please check your input values and model compatibility.")
                    finally:
                        progress_bar.empty()

        # Information sections
        st.markdown("---")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            with st.expander("🔬 About CO₂/N₂ Selectivity"):
                st.markdown("""
                **CO₂/N₂ Selectivity** is crucial for gas separation membranes, representing the ability to preferentially allow CO₂ passage while blocking N₂.
                
                - **Higher selectivity** = Better separation efficiency
                - **Industrial targets**: 20-50+ typically
                - **Applications**: 
                  - Carbon capture and storage
                  - Natural gas purification
                  - Air separation processes
                  - Environmental remediation
                """)

        with col_info2:
            with st.expander("📊 Model Details"):
                st.markdown("""
                Our Random Forest model was trained on comprehensive polymer membrane data.
                
                **Training Dataset:**
                - **Samples:** 1,407 polymer membranes
                - **Features:** 63 input parameters
                - **Cross-validation:** 5-fold CV
                
                **Performance Metrics:**
                - **R² Score:** 0.9175 (excellent)
                - **MAE:** 1.76 (low error)
                - **RMSE:** 3.11 (robust predictions)
                
                **Algorithm:** Random Forest Regressor with optimized hyperparameters
                """)

        with col_info3:
            with st.expander("🔧 Technical Information"):
                st.markdown(f"""
                **App Details:**
                - **Framework:** Streamlit
                - **ML Library:** Scikit-learn {st.session_state.get('sklearn_version', 'Unknown')}
                - **Model File:** {model_file or 'Not loaded'}
                - **Version:** 2.0.0
                - **Developer:** Enhanced by Claude
                
                **Features:**
                - Real-time ML predictions
                - Interactive visualizations
                - Comprehensive input validation
                - Performance comparison tools
                - Enhanced error handling
                - Modern responsive UI
                """)

    else:
        # Show comprehensive troubleshooting guide when model fails to load
        st.error("## 🚨 Model Loading Failed")
        
        tab1, tab2, tab3 = st.tabs(["🔧 Quick Fix", "📋 Diagnostics", "🛠️ Advanced"])
        
        with tab1:
            st.markdown("""
            ### 🎯 Most Common Solutions:
            
            1. **Check file names** - Rename your model file to one of these:
               ```
               PolyMemCO2Pipeline.joblib
               PolyMemCO2Pipeline.pkl
               model.joblib
               model.pkl
               ```
            
            2. **Update sklearn version** in requirements.txt:
               ```
               scikit-learn==1.3.2
               ```
            
            3. **Retrain your model** with current sklearn version if compatibility issues persist.
            """)
        
        with tab2:
            st.markdown("### 🔍 System Diagnostics")
            
            # Show current environment info
            try:
                import sklearn
                st.success(f"✅ **sklearn version:** {sklearn.__version__}")
            except ImportError:
                st.error("❌ **sklearn not available** - Install with `pip install scikit-learn`")
            
            # Show available files
            current_files = [f for f in os.listdir('.') if f.endswith(('.joblib', '.pkl'))]
            if current_files:
                st.info(f"📁 **Found files:** {', '.join(current_files)}")
            else:
                st.warning("⚠️ **No .joblib/.pkl files found**")
                
            # Show directory contents
            with st.expander("📂 Directory Contents"):
                all_files = os.listdir('.')
                st.write(all_files)
        
        with tab3:
            st.markdown("""
            ### 🛠️ Advanced Troubleshooting
            
            **Model Version Compatibility:**
            ```python
            # When saving your model, ensure compatibility:
            import joblib
            import pickle
            
            # Method 1: Use joblib (recommended)
            joblib.dump(your_model, 'PolyMemCO2Pipeline.joblib')
            
            # Method 2: Use pickle (fallback)
            with open('PolyMemCO2Pipeline.pkl', 'wb') as f:
                pickle.dump(your_model, f)
            ```
            
            **Custom Transformer Issues:**
            - Ensure `RarePolymerGrouper` class matches training version
            - Consider using `sklearn.compose.ColumnTransformer` for better compatibility
            
            **Environment Setup:**
            ```bash
            pip install streamlit scikit-learn==1.3.2 pandas numpy plotly
            ```
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4>🚀 Enhanced CO₂/N₂ Selectivity Predictor</h4>
        <p><em>Developed with ❤️ using Streamlit, Scikit-learn & Plotly</em></p>
        <p><strong>For educational and research purposes only</strong></p>
        <small>Version 2.0.0 | Enhanced UI & Error Handling</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
