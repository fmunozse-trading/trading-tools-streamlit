import streamlit as st

def apply_global_styles():
    """
    Aplica estilos CSS globales para reducir el tamaño de la fuente
    y hacer la interfaz más compacta.
    """
    st.markdown("""
        <style>
        /* Ajuste global del tamaño de fuente base */
        html {
            font-size: 14px; /* Default es 16px, esto reduce todo ~12.5% */
        }
        
        /* Forzar herencia en contenedores de Streamlit */
        .stApp {
            font-size: 14px;
        }
        
        /* Títulos más compactos */
        h1 { font-size: 1.8rem !important; } /* Antes ~2.5rem */
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.25rem !important; }
        
        /* Ajustar tamaño de widgets comunes */
        .stButton button {
            font-size: 14px !important;
        }
        
        .stSelectbox div[data-baseweb="select"] div {
            font-size: 14px !important;
        }
        
        /* Métricas más compactas pero legibles */
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }
        
        /* Ajustar espaciados para reducir altura */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
