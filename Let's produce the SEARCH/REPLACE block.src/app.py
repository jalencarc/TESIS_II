import streamlit as st
import joblib
import os

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Tesis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Load models (with error handling)
# ----------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

models = {}
model_names = {
    "rf_model.pkl": "Random Forest",
    "gb_model.pkl": "Gradient Boosting",
    "xgb_model.pkl": "XGBoost",
}

for filename, display_name in model_names.items():
    path = os.path.join(MODELS_DIR, filename)
    try:
        models[display_name] = joblib.load(path)
    except Exception as e:
        st.error(f"No se pudo cargar {filename}: {e}")
        models[display_name] = None

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Ir a:",
    ["Evaluación de Modelos", "Simulador de Predicción"],
)

# ----------------------------------------------------------------------
# Page content
# ----------------------------------------------------------------------
if page == "Evaluación de Modelos":
    st.header("Evaluación de Modelos")
    st.markdown(
        "Aquí se mostrarán las métricas de rendimiento de los modelos "
        "cargados (Random Forest, Gradient Boosting, XGBoost)."
    )

    # Show which models were loaded successfully
    for name, model in models.items():
        if model is not None:
            st.success(f"✅ {name} cargado correctamente.")
        else:
            st.warning(f"⚠️ {name} no disponible.")

elif page == "Simulador de Predicción":
    st.header("Simulador de Predicción")
    st.markdown(
        "En esta sección se podrán ingresar variables de entrada "
        "y obtener predicciones de los modelos."
    )

    # Placeholder for future input fields
    st.info("Funcionalidad en desarrollo.")
