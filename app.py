import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ Always first Streamlit command
st.set_page_config(page_title="CO₂/N₂ Selectivity Predictor", layout="centered")

# 🧠 Define RarePolymerGrouper so joblib can load it
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

# --- Load pipeline ---
@st.cache_resource
def load_pipeline():
    return joblib.load("PolyMemCO2Pipeline.joblib")

model = load_pipeline()

# --- App Title ---
st.title("🧪 CO₂/N₂ Selectivity Prediction App")
st.markdown("Predict CO₂/N₂ Selectivity using polymer membrane properties")

# --- User Input ---
st.header("📥 Input Parameters")
with st.form("input_form"):
    
    # Polymer Type selectbox with predefined options
    polymer_type_options = [
        "Polyimides and Polypyrrolones",
        "Polymers with High Free Volume", 
        "Thermally Rearranged Polymers",
        "Silicone rubber and variants",
        "Substituted Polyacetylenes",
        "Mixed Matrix Membranes",
        "Other"
    ]
    
    polymer_type_choice = st.selectbox("Polymer Type", polymer_type_options, index=0)
    
    # If "Other" is selected, show text input
    if polymer_type_choice == "Other":
        polymer_type = st.text_input("Enter custom Polymer Type:", value="")
    else:
        polymer_type = polymer_type_choice
    
    # Polymer selectbox with predefined options
    polymer_options = [
        "Polyimide",
        "Polysulfone", 
        "Polyetherimide",
        "Poly(vinyl chloride)",
        "Polytrimethylsilylpropyne",
        "Polypyrrolone",
        "Cellulose acetate",
        "Polycarbonate",
        "Polyethersulfone",
        "Other"
    ]
    
    polymer_choice = st.selectbox("Polymer", polymer_options, index=0)
    
    # If "Other" is selected, show text input
    if polymer_choice == "Other":
        polymer = st.text_input("Enter custom Polymer:", value="")
    else:
        polymer = polymer_choice

    # Gas permeability inputs
    st.subheader("Gas Permeability Values")
    
    # Organize inputs in columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        He_Permeability = st.number_input("He Permeability", min_value=0.0, value=0.0)
        H2_Permeability = st.number_input("H₂ Permeability", min_value=0.0, value=0.0)
        O2_Permeability = st.number_input("O₂ Permeability", min_value=0.0, value=0.0)
        N2_Permeability = st.number_input("N₂ Permeability", min_value=0.0, value=0.0)
        CO2_Permeability = st.number_input("CO₂ Permeability", min_value=0.0, value=0.0)
    
    with col2:
        CH4_Permeability = st.number_input("CH₄ Permeability", min_value=0.0, value=0.0)
        C2H4_Permeability = st.number_input("C₂H₄ Permeability", min_value=0.0, value=0.0)
        C2H6_Permeability = st.number_input("C₂H₆ Permeability", min_value=0.0, value=0.0)
        C3H6_Permeability = st.number_input("C₃H₆ Permeability", min_value=0.0, value=0.0)
        C3H8_Permeability = st.number_input("C₃H₈ Permeability", min_value=0.0, value=0.0)
    
    with col3:
        C4H8_Permeability = st.number_input("C₄H₈ Permeability", min_value=0.0, value=0.0)
        nC4H10_Permeability = st.number_input("n-C₄H₁₀ Permeability", min_value=0.0, value=0.0)
        CF4_Permeability = st.number_input("CF₄ Permeability", min_value=0.0, value=0.0)
        C2F6_Permeability = st.number_input("C₂F₆ Permeability", min_value=0.0, value=0.0)
        C3F8_Permeability = st.number_input("C₃F₈ Permeability", min_value=0.0, value=0.0)

    submitted = st.form_submit_button("🔮 Predict")

# --- Prediction Logic ---
if submitted:
    try:
        # Validate inputs
        if polymer_type_choice == "Other" and not polymer_type:
            st.error("Please enter a custom Polymer Type or select a predefined option.")
        elif polymer_choice == "Other" and not polymer:
            st.error("Please enter a custom Polymer or select a predefined option.")
        else:
            # Create input DataFrame in the same format
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

            # Predict using pipeline
            prediction = model.predict(input_df)[0]

            # Display result
            st.success(f"🌟 Predicted CO₂/N₂ Selectivity: **{prediction:.2f}**")
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                st.write(f"**Polymer Type:** {polymer_type}")
                st.write(f"**Polymer:** {polymer}")
                st.write("**Non-zero Permeability Values:**")
                for key, value in input_data.items():
                    if key not in ["Polymer Type", "Polymer"] and value[0] > 0:
                        st.write(f"- {key}: {value[0]}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.error("Please check your inputs and try again.")

# --- Sidebar with Information ---
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This app predicts CO₂/N₂ selectivity for polymer membranes based on:
- Polymer type and specific polymer
- Gas permeability values for various gases

**Instructions:**
1. Select or enter polymer type and polymer name
2. Enter permeability values (leave as 0 if unknown)
3. Click Predict to get selectivity prediction
""")

st.sidebar.header("📊 Typical Values")
st.sidebar.markdown("""
**Common Permeability Ranges:**
- He: 1-100 Barrer
- H₂: 1-50 Barrer  
- CO₂: 1-200 Barrer
- O₂: 0.1-20 Barrer
- N₂: 0.01-5 Barrer
""")

# Note: 1 Barrer = 10⁻¹⁰ cm³(STP)·cm/(cm²·s·cmHg)
st.sidebar.caption("Units: Barrer (10⁻¹⁰ cm³(STP)·cm/(cm²·s·cmHg))")