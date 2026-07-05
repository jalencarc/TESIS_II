import streamlit as st
import joblib
import os
import io
import unicodedata
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ══════════════════════════════════════════════════════════════
# CONFIGURACION DE PAGINA
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sistema de Prediccion de Incidentes Laborales",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# ESTILOS CSS PROFESIONALES
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .pred-leve {
        background: #E3F2FD;
        border: 2px solid #1565C0;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        color: #0D47A1;
    }
    .pred-grave {
        background: #FFF3E0;
        border: 2px solid #E65100;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        color: #E65100;
    }
    .pred-mortal {
        background: #FFEBEE;
        border: 2px solid #B71C1C;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        color: #B71C1C;
    }
    .badge-ok {
        background: #E8F5E9;
        border: 1.5px solid #2E7D32;
        border-radius: 8px;
        padding: .6rem 1rem;
        display: inline-block;
        margin-bottom: .4rem;
        color: #1B5E20;
    }
    .demo-box {
        background: #F3E5F5;
        border: 2px solid #6A1B9A;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #4A148C;
    }
    .section-title {
        color: #1A237E;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTES Y RUTAS
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

REQUIRED_COLS = ["SECTOR", "SEXO", "OCUPACION", "TIPO_ACCIDENTE"]
MODEL_NAMES = ["Random Forest", "Gradient Boosting", "XGBoost"]
CLASS_LABELS = ["Leve (1)", "Grave (2)", "Mortal (3)"]
COLORS_PALETTE = ["#1A237E", "#1565C0", "#42A5F5"]

SEV_LABEL_MAP = {1: "LEVE", 2: "GRAVE", 3: "MORTAL"}
SEV_CSS_MAP = {"LEVE": "pred-leve", "GRAVE": "pred-grave", "MORTAL": "pred-mortal"}
SEV_EMOJI_MAP = {"LEVE": "🟢", "GRAVE": "🟠", "MORTAL": "🔴"}
SEV_TEXT_COLOR = {"LEVE": "#0D47A1", "GRAVE": "#E65100", "MORTAL": "#B71C1C"}

# ══════════════════════════════════════════════════════════════
# CARGA CACHEADA DE MODELOS Y ENCODERS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_modelos():
    resultado = {"models": {}, "encoders": None, "errors": []}

    enc_path = os.path.join(MODELS_DIR, "encoders.pkl")
    if os.path.exists(enc_path):
        try:
            resultado["encoders"] = joblib.load(enc_path)
        except Exception as e:
            resultado["errors"].append(f"encoders.pkl: {e}")
    else:
        resultado["errors"].append("encoders.pkl no encontrado en /models")

    model_files = {
        "Random Forest":     "rf_model.pkl",
        "Gradient Boosting": "gb_model.pkl",
        "XGBoost":           "xgb_model.pkl",
    }
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                resultado["models"][name] = joblib.load(path)
            except Exception as e:
                resultado["errors"].append(f"{fname}: {e}")
        else:
            resultado["errors"].append(f"{fname} no encontrado")

    return resultado

carga = cargar_modelos()
ENCODERS = carga["encoders"]
MODELS = carga["models"]
LOAD_ERRORS = carga["errors"]

# ══════════════════════════════════════════════════════════════
# METRICAS DE ENTRENAMIENTO (HARDCODEADAS - VALORES REALES)
# ══════════════════════════════════════════════════════════════
METRICAS = {
    "Random Forest": {
        "accuracy": 74.46,
        "cv_mean":  73.33,
        "cv_std":   0.65,
        "precision": [67.82, 75.56, 80.88],
        "recall":    [66.13, 69.48, 96.99],
        "f1":        [66.97, 72.39, 88.20],
        "cm":        [[529, 224, 47], [251, 742, 75], [0, 16, 516]],
        "feat_imp":  [55.4, 21.3, 17.0, 6.1],
        "color":     "#1565C0",
    },
    "Gradient Boosting": {
        "accuracy": 72.63,
        "cv_mean":  72.04,
        "cv_std":   0.98,
        "precision": [67.39, 73.02, 78.50],
        "recall":    [62.25, 73.50, 86.47],
        "f1":        [64.74, 73.27, 82.31],
        "cm":        [[498, 258, 44], [201, 785, 82], [40, 32, 460]],
        "feat_imp":  [],
        "color":     "#2E7D32",
    },
    "XGBoost": {
        "accuracy": 70.75,
        "cv_mean":  70.60,
        "cv_std":   0.96,
        "precision": [69.37, 70.69, 72.41],
        "recall":    [57.75, 75.19, 81.39],
        "f1":        [63.06, 72.87, 76.60],
        "cm":        [[462, 274, 64], [164, 803, 101], [40, 59, 433]],
        "feat_imp":  [],
        "color":     "#E65100",
    },
}

MEJOR_MODELO = "Random Forest"
FEATURE_NAMES = ["TIPO_ACCIDENTE", "SECTOR", "OCUPACION", "SEXO"]

# ══════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════

def normalizar_texto(texto):
    """Normaliza texto: remueve tildes, dieresis, pasa a minusculas y elimina espacios extra."""
    s = str(texto)
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = s.lower().strip()
    s = " ".join(s.split())
    return s


def etiqueta_severidad(pred):
    """Convierte prediccion numerica (1-3) a etiqueta legible."""
    return SEV_LABEL_MAP.get(int(pred), "?")


def codificar_fila(sector, sexo, ocupacion, tipo_accidente):
    """Codifica una fila de entrada usando los encoders cargados."""
    return np.array([[
        ENCODERS["SECTOR"].transform([sector])[0],
        ENCODERS["SEXO"].transform([sexo])[0],
        ENCODERS["OCUPACION"].transform([ocupacion])[0],
        ENCODERS["TIPO_ACCIDENTE"].transform([tipo_accidente])[0],
    ]])


def tarjeta_prediccion_html(label, model_name, probas):
    """Genera HTML de tarjeta de resultado con colores explicitos."""
    tc = SEV_TEXT_COLOR[label]
    em = SEV_EMOJI_MAP[label]
    css = SEV_CSS_MAP[label]
    return (
        f'<div class="{css}">'
        f'<p style="margin:0;font-size:.9rem;color:{tc}">{model_name}</p>'
        f'<h1 style="color:{tc}">{em}</h1>'
        f'<h2 style="color:{tc}"><b>{label}</b></h2>'
        f'<p style="color:{tc}">Leve: {probas[0]*100:.1f}% &middot; '
        f'Grave: {probas[1]*100:.1f}% &middot; Mortal: {probas[2]*100:.1f}%</p>'
        f'</div>'
    )


def predecir_caso(model_name, X):
    """Predice un caso individual con el modelo especificado.
    Retorna (prediccion, probabilidades).
    XGBoost: se suma +1 al resultado (clases originales 0-2 -> severidad 1-3).
    """
    obj = MODELS[model_name]
    pred = obj.predict(X)[0]
    if model_name == "XGBoost":
        pred = pred + 1
    proba = obj.predict_proba(X)[0]
    return pred, proba


def predecir_lote(model_name, df, col_map):
    """Predice un lote completo de registros.
    Retorna (predicciones, probabilidades, advertencias).
    Valida valores contra el encoder y usa fallback si no existen.
    """
    obj = MODELS[model_name]
    cols_internas = REQUIRED_COLS
    filas = []
    advertencias = []

    for idx, row in df.iterrows():
        vec = []
        for col_int in cols_internas:
            user_col = col_map.get(col_int)
            if user_col is None or user_col not in df.columns:
                vec.append(0)
                continue
            val = str(row[user_col])
            enc = ENCODERS.get(col_int)
            if enc is None:
                vec.append(0)
                continue
            try:
                vec.append(enc.transform([val])[0])
            except (ValueError, KeyError):
                fallback = enc.classes_[0]
                advertencias.append(
                    f"Fila {idx+2}: '{val}' ({col_int}) no encontrado en encoder. "
                    f"Usando fallback '{fallback}'."
                )
                vec.append(0)
        filas.append(vec)

    X = np.array(filas)
    preds = obj.predict(X)
    if model_name == "XGBoost":
        preds = preds + 1

    probas = None
    try:
        probas = obj.predict_proba(X)
    except Exception:
        pass

    return preds, probas, advertencias


def detectar_mapeo_columnas(df_columnas, cols_requeridas):
    """Deteccion inteligente de columnas por match exacto normalizado.
    Retorna (mapeo, no_mapeadas, todas_detectadas).
    """
    norm_requeridas = {normalizar_texto(c): c for c in cols_requeridas}
    norm_columnas = {normalizar_texto(c): c for c in df_columnas}

    mapeo = {}
    no_mapeadas = []

    for col_req in cols_requeridas:
        n = normalizar_texto(col_req)
        if n in norm_columnas:
            mapeo[col_req] = norm_columnas[n]
        else:
            no_mapeadas.append(col_req)

    todas = len(no_mapeadas) == 0
    return mapeo, no_mapeadas, todas


def leer_archivo_subido(archivo):
    """Lee CSV con deteccion automatica de separador o Excel con selector de hoja."""
    nombre = archivo.name.lower()

    if nombre.endswith(".csv"):
        contenido = archivo.read()
        archivo.seek(0)
        try:
            df = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(contenido))
            except Exception as e:
                st.error(f"Error al leer el CSV: {e}")
                return None, None
        return df, "CSV"

    elif nombre.endswith((".xlsx", ".xls")):
        try:
            xl = pd.ExcelFile(archivo)
        except Exception as e:
            st.error(f"Error al leer el archivo Excel: {e}")
            return None, None

        if len(xl.sheet_names) > 1:
            hoja = st.selectbox(
                "Selecciona la hoja del Excel:",
                xl.sheet_names,
                key="sheet_selector",
            )
        else:
            hoja = xl.sheet_names[0]

        df = xl.parse(hoja)
        return df, f"Excel › {hoja}"

    else:
        st.error("Formato no soportado. Usa archivos .csv, .xlsx o .xls")
        return None, None


def generar_excel_descarga(df):
    """Genera un archivo Excel en memoria usando openpyxl."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predicciones")
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🦺 Navegacion")
pagina = st.sidebar.radio("", [
    "🏠 Inicio",
    "🎓 Demo para Docente",
    "📊 Evaluacion de Modelos",
    "🔮 Prediccion con Nuevos Datos",
])
st.sidebar.markdown("---")
st.sidebar.markdown("### Estado del sistema")

if ENCODERS is not None:
    st.sidebar.markdown('<div class="badge-ok">✅ Encoders cargados</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown("❌ Encoders no disponibles")

for mn in MODEL_NAMES:
    if mn in MODELS:
        st.sidebar.markdown(
            f'<div class="badge-ok">✅ {mn}</div>', unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(f"❌ {mn}")

if LOAD_ERRORS:
    with st.sidebar.expander("⚠️ Errores de carga"):
        for e in LOAD_ERRORS:
            st.markdown(f"- {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Tesis:** Uso de Machine Learning para el Analisis de Incidentes Laborales en Lima Metropolitana")
st.sidebar.markdown("**Autor:** Jorge Enrique Alencar Coral")
st.sidebar.markdown("**Universidad:** Cesar Vallejo")
st.sidebar.markdown("**Dataset:** 12,000 registros MTPE")

# ══════════════════════════════════════════════════════════════
# PAGINA 1: INICIO
# ══════════════════════════════════════════════════════════════
if pagina == "🏠 Inicio":
    st.markdown(
        "<h1 style='text-align:center;color:#1A237E'>"
        "🦺 Sistema de Prediccion de Severidad de Incidentes Laborales"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#555;font-size:1.05rem'>"
        "Lima Metropolitana &middot; Datos MTPE &middot; 12,000 registros "
        "&middot; 3 algoritmos de ensamble supervisado"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🥇 Mejor Modelo",  MEJOR_MODELO)
    c2.metric("🎯 Exactitud",     f"{METRICAS[MEJOR_MODELO]['accuracy']} %")
    c3.metric("⚠️ Recall Mortal", f"{METRICAS[MEJOR_MODELO]['recall'][2]} %")
    c4.metric("📋 Registros",     "12,000")

    st.markdown("---")
    st.markdown("### Que hace este sistema?")

    a, b = st.columns(2)
    with a:
        st.info(
            "**📊 Evaluacion**\n\n"
            "Muestra metricas reales de los 3 modelos entrenados: "
            "accuracy, precision, recall, F1 y matrices de confusion."
        )
        st.info(
            "**🔮 Prediccion en lote**\n\n"
            "Carga CSV o Excel con nuevos registros y obtiene "
            "la severidad predicha para cada uno."
        )
    with b:
        st.info(
            "**🎓 Demo para Docente**\n\n"
            "Seccion dedicada a demostrar que los modelos estan "
            "entrenados y funcionando con datos reales."
        )
        st.info(
            "**⬇️ Descarga de resultados**\n\n"
            "Exporta predicciones en CSV o Excel con probabilidades "
            "por clase incluidas."
        )

    st.markdown("---")
    st.markdown(
        "### Flujo de uso\n"
        "1. Ve a **🎓 Demo para Docente** para una demostracion rapida e impactante\n"
        "2. Explora **📊 Evaluacion de Modelos** para ver todas las metricas de entrenamiento\n"
        "3. Usa **🔮 Prediccion con Nuevos Datos** para cargar tu propio CSV o Excel"
    )

# ══════════════════════════════════════════════════════════════
# PAGINA 2: DEMO PARA DOCENTE
# ══════════════════════════════════════════════════════════════
elif pagina == "🎓 Demo para Docente":
    st.markdown("## 🎓 Demostracion para Evaluacion Docente")
    st.markdown(
        '<div class="demo-box">'
        "<b>Esta seccion demuestra que los 3 modelos estan correctamente "
        "entrenados y en memoria, y pueden realizar predicciones en tiempo real "
        "sobre datos que nunca han visto.</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not MODELS or ENCODERS is None:
        st.error("❌ No se pudieron cargar los modelos. Verifica que los .pkl esten en /models.")
        st.stop()

    # 1. Modelos cargados en memoria
    st.markdown("### ✅ 1. Modelos cargados en memoria")
    for mn, obj in MODELS.items():
        nf = getattr(obj, "n_features_in_", 4)
        nc = len(obj.classes_) if hasattr(obj, "classes_") else "N/A"
        acc = METRICAS[mn]["accuracy"]
        st.markdown(
            f'<div class="badge-ok">'
            f"🤖 <b>{mn}</b> &nbsp;|&nbsp; "
            f"Clases: {nc} &nbsp;|&nbsp; "
            f"Features: {nf} &nbsp;|&nbsp; "
            f"Accuracy: <b>{acc}%</b>"
            f"</div><br>",
            unsafe_allow_html=True,
        )

    # 2. Prediccion en vivo con los 3 modelos
    st.markdown("---")
    st.markdown("### 🔴 2. Prediccion en vivo — los 3 modelos simultaneamente")
    st.markdown(
        '<p style="color:#555">Selecciona un caso real y observa como los 3 modelos '
        'predicen al mismo tiempo:</p>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2)
    with d1:
        demo_sector = st.selectbox(
            "Sector economico:",
            sorted(ENCODERS["SECTOR"].classes_),
            key="demo_sec",
        )
        demo_sexo = st.selectbox(
            "Sexo:",
            sorted(ENCODERS["SEXO"].classes_),
            key="demo_sex",
        )
    with d2:
        demo_ocupacion = st.selectbox(
            "Ocupacion:",
            sorted(ENCODERS["OCUPACION"].classes_),
            key="demo_ocu",
        )
        demo_tipo = st.selectbox(
            "Tipo de accidente:",
            sorted(ENCODERS["TIPO_ACCIDENTE"].classes_),
            key="demo_tip",
        )

    if st.button("🚀 Predecir con los 3 modelos", type="primary"):
        X_demo = codificar_fila(demo_sector, demo_sexo, demo_ocupacion, demo_tipo)
        columnas = st.columns(3)
        for i, (mn, obj) in enumerate(MODELS.items()):
            pred, proba = predecir_caso(mn, X_demo)
            label = etiqueta_severidad(pred)
            with columnas[i]:
                st.markdown(
                    tarjeta_prediccion_html(label, mn, proba),
                    unsafe_allow_html=True,
                )
        st.markdown("---")
        st.success("✅ Los 3 modelos realizaron predicciones correctamente con el mismo caso de entrada.")

    # 3. Tabla comparativa de metricas
    st.markdown("---")
    st.markdown("### 📈 3. Resumen comparativo de rendimiento")
    df_comp = pd.DataFrame({
        "Modelo":             list(METRICAS.keys()),
        "Accuracy (%)":       [m["accuracy"]   for m in METRICAS.values()],
        "CV Media (%)":       [m["cv_mean"]     for m in METRICAS.values()],
        "Recall Mortal (%)":  [m["recall"][2]   for m in METRICAS.values()],
        "F1 Mortal (%)":      [m["f1"][2]       for m in METRICAS.values()],
    })
    st.dataframe(
        df_comp.style
        .highlight_max(
            subset=["Accuracy (%)", "CV Media (%)", "Recall Mortal (%)", "F1 Mortal (%)"],
            color="#C8E6C9",
        )
        .format({
            "Accuracy (%)":      "{:.2f}",
            "CV Media (%)":      "{:.2f}",
            "Recall Mortal (%)": "{:.2f}",
            "F1 Mortal (%)":     "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "💡 **Random Forest** obtiene el mejor accuracy general y el mayor Recall "
        "para casos Mortales (96.99%), lo cual es critico en seguridad laboral."
    )

# ══════════════════════════════════════════════════════════════
# PAGINA 3: EVALUACION DE MODELOS
# ══════════════════════════════════════════════════════════════
elif pagina == "📊 Evaluacion de Modelos":
    st.markdown("## 📊 Evaluacion de Modelos")
    modelo_sel = st.selectbox("Selecciona un modelo:", list(METRICAS.keys()))
    m = METRICAS[modelo_sel]
    st.markdown("---")

    ce1, ce2, ce3 = st.columns(3)
    ce1.metric("Exactitud (Accuracy)", f"{m['accuracy']} %")
    ce2.metric("CV 5-Fold Media",      f"{m['cv_mean']} %")
    ce3.metric("CV Desv. Estandar",    f"± {m['cv_std']} %")
    st.markdown("---")

    cl, cr = st.columns(2)

    # Grafico de barras: Precision, Sensibilidad, F1
    with cl:
        fig = go.Figure()
        for i, (nombre, valores) in enumerate({
            "Precision":    m["precision"],
            "Sensibilidad": m["recall"],
            "F1-Score":     m["f1"],
        }.items()):
            fig.add_trace(go.Bar(
                name=nombre,
                x=CLASS_LABELS,
                y=valores,
                marker_color=COLORS_PALETTE[i],
                text=[f"{v}%" for v in valores],
                textposition="outside",
            ))
        fig.update_layout(
            title=dict(text=f"Metricas por Clase — {modelo_sel}", font=dict(color="#1A237E")),
            barmode="group",
            yaxis=dict(range=[40, 110], title="Porcentaje (%)"),
            height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Matriz de confusion
    with cr:
        fig_cm = px.imshow(
            m["cm"],
            labels=dict(x="Predicho", y="Real", color="Casos"),
            x=CLASS_LABELS,
            y=CLASS_LABELS,
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(
            title=dict(text=f"Matriz de Confusion — {modelo_sel}", font=dict(color="#1A237E")),
            height=420,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Importancia de variables (solo Random Forest)
    if modelo_sel == "Random Forest" and m["feat_imp"]:
        st.markdown("---")
        valores_imp = m["feat_imp"]
        nombres_imp = FEATURE_NAMES
        pares = sorted(zip(valores_imp, nombres_imp))

        fig_fi = go.Figure(go.Bar(
            x=[p[0] for p in pares],
            y=[p[1] for p in pares],
            orientation="h",
            marker_color=["#90CAF9", "#42A5F5", "#1565C0", "#1A237E"],
            text=[f"{p[0]}%" for p in pares],
            textposition="outside",
        ))
        fig_fi.update_layout(
            title=dict(text="Importancia de Variables — Random Forest", font=dict(color="#1A237E")),
            xaxis=dict(range=[0, 65], title="Importancia (%)"),
            height=360,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Comparativa general con CV
    st.markdown("---")
    st.markdown("### Comparativa General")
    algs = list(METRICAS.keys())
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Accuracy (Test)",
        x=algs,
        y=[METRICAS[a]["accuracy"] for a in algs],
        marker_color=[METRICAS[a]["color"] for a in algs],
        text=[f"{METRICAS[a]['accuracy']}%" for a in algs],
        textposition="outside",
    ))
    fig_comp.add_trace(go.Scatter(
        name="CV 5-Fold Media",
        x=algs,
        y=[METRICAS[a]["cv_mean"] for a in algs],
        mode="markers+lines",
        marker=dict(size=14, color="#B71C1C", symbol="diamond"),
        error_y=dict(
            type="data",
            array=[METRICAS[a]["cv_std"] for a in algs],
            visible=True,
            color="#B71C1C",
        ),
        line=dict(dash="dash", color="#B71C1C", width=2),
    ))
    fig_comp.update_layout(
        title=dict(text="Exactitud y Validacion Cruzada — Comparativa", font=dict(color="#1A237E")),
        yaxis=dict(range=[65, 80], title="Porcentaje (%)"),
        height=440,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGINA 4: PREDICCION CON NUEVOS DATOS
# ══════════════════════════════════════════════════════════════
elif pagina == "🔮 Prediccion con Nuevos Datos":
    st.markdown("## 🔮 Prediccion con Nuevos Datos")

    if ENCODERS is None:
        st.error("❌ Encoders no disponibles. Verifica /models/encoders.pkl")
        st.stop()
    if not MODELS:
        st.error("❌ No hay modelos cargados. Verifica /models/*.pkl")
        st.stop()

    modo = st.radio("Modo de entrada:", ["📝 Caso manual", "📂 CSV / Excel (lote)"])
    st.markdown("---")

    # ─────────────────────────────────────
    # MODO MANUAL
    # ─────────────────────────────────────
    if modo == "📝 Caso manual":
        c1, c2 = st.columns(2)
        with c1:
            manual_sector = st.selectbox(
                "Sector economico:",
                sorted(ENCODERS["SECTOR"].classes_),
                key="man_sec",
            )
            manual_sexo = st.selectbox(
                "Sexo:",
                sorted(ENCODERS["SEXO"].classes_),
                key="man_sex",
            )
        with c2:
            manual_ocupacion = st.selectbox(
                "Ocupacion:",
                sorted(ENCODERS["OCUPACION"].classes_),
                key="man_ocu",
            )
            manual_tipo = st.selectbox(
                "Tipo de accidente:",
                sorted(ENCODERS["TIPO_ACCIDENTE"].classes_),
                key="man_tip",
            )

        manual_modelo = st.selectbox("Modelo a usar:", list(MODELS.keys()))
        st.markdown("---")

        if st.button("🔍 Predecir", type="primary"):
            X_manual = codificar_fila(manual_sector, manual_sexo, manual_ocupacion, manual_tipo)
            pred_manual, proba_manual = predecir_caso(manual_modelo, X_manual)
            label_manual = etiqueta_severidad(pred_manual)

            cr_man, cp_man = st.columns(2)
            with cr_man:
                st.markdown(
                    tarjeta_prediccion_html(label_manual, manual_modelo, proba_manual),
                    unsafe_allow_html=True,
                )
            with cp_man:
                fig_p = go.Figure(go.Bar(
                    x=[p * 100 for p in proba_manual],
                    y=CLASS_LABELS,
                    orientation="h",
                    marker_color=COLORS_PALETTE,
                    text=[f"{p*100:.1f}%" for p in proba_manual],
                    textposition="outside",
                ))
                fig_p.update_layout(
                    title=dict(text="Probabilidad por Clase", font=dict(color="#1A237E")),
                    xaxis=dict(range=[0, 120], title="Probabilidad (%)"),
                    height=300,
                )
                st.plotly_chart(fig_p, use_container_width=True)

            st.dataframe(
                pd.DataFrame({
                    "Variable": [
                        "Sector", "Sexo", "Ocupacion",
                        "Tipo de Accidente", "Severidad Predicha", "Modelo",
                    ],
                    "Valor": [
                        manual_sector, manual_sexo, manual_ocupacion,
                        manual_tipo, label_manual, manual_modelo,
                    ],
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ─────────────────────────────────────
    # MODO LOTE (CSV / EXCEL)
    # ─────────────────────────────────────
    else:
        st.markdown("### Cargar archivo con nuevos registros")
        st.markdown(
            '<p style="color:#555">Sube tu archivo. El sistema detectara automaticamente '
            'las columnas requeridas.</p>',
            unsafe_allow_html=True,
        )

        archivo_subido = st.file_uploader(
            "Arrastra o selecciona un archivo (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
        )

        if archivo_subido:
            df_nuevo, fmt_desc = leer_archivo_subido(archivo_subido)
            if df_nuevo is None:
                st.stop()
            if len(df_nuevo) == 0:
                st.error("❌ El archivo esta vacio. Sube un archivo con registros.")
                st.stop()

            st.markdown(f"**{fmt_desc}** &middot; {len(df_nuevo):,} filas &middot; {len(df_nuevo.columns)} columnas")
            st.dataframe(df_nuevo.head(8), use_container_width=True)

            # --- MAPEO INTELIGENTE DE COLUMNAS ---
            st.markdown("---")
            st.markdown("### Mapeo de columnas")

            mapeo_auto, no_mapeadas, todas_detectadas = detectar_mapeo_columnas(
                df_nuevo.columns, REQUIRED_COLS
            )

            if todas_detectadas:
                st.success(
                    "✅ Las 4 columnas requeridas fueron detectadas automaticamente:\n\n"
                    + "  \n".join(f"**{k}** → `{v}`" for k, v in mapeo_auto.items())
                )
                col_map = mapeo_auto
            else:
                st.info(
                    f"Se detectaron **{len(mapeo_auto)} de 4** columnas automaticamente. "
                    "Asigna las columnas faltantes:"
                )
                if mapeo_auto:
                    st.markdown("**Columnas detectadas:** " + ", ".join(
                        f"{k} → `{v}`" for k, v in mapeo_auto.items()
                    ))

                opciones = ["(ninguna)"] + list(df_nuevo.columns)
                col_map = dict(mapeo_auto)

                for col_faltante in no_mapeadas:
                    idx_default = 0
                    col_map[col_faltante] = st.selectbox(
                        f"Columna para **{col_faltante}**:",
                        opciones,
                        index=idx_default,
                        key=f"map_{col_faltante}",
                    )

            # Validar que todas las columnas tienen asignacion
            sin_asignar = [k for k in REQUIRED_COLS if col_map.get(k, "(ninguna)") == "(ninguna)"]
            if sin_asignar:
                st.warning(
                    f"⚠️ Sin columna asignada para: {', '.join(sin_asignar)}. "
                    "Se usara valor 0 por defecto."
                )

            st.markdown("---")
            modelo_lote = st.selectbox("Modelo a usar:", list(MODELS.keys()))

            if st.button("🔍 Predecir todos los registros", type="primary"):
                with st.spinner(f"Procesando {len(df_nuevo):,} registros con {modelo_lote}..."):
                    preds_lote, probas_lote, warnings_list = predecir_lote(
                        modelo_lote, df_nuevo, col_map
                    )

                df_result = df_nuevo.copy()
                df_result["SEVERIDAD_PREDICHA"] = [etiqueta_severidad(p) for p in preds_lote]

                if probas_lote is not None:
                    df_result["PROB_LEVE (%)"]   = (probas_lote[:, 0] * 100).round(1)
                    df_result["PROB_GRAVE (%)"]  = (probas_lote[:, 1] * 100).round(1)
                    df_result["PROB_MORTAL (%)"] = (probas_lote[:, 2] * 100).round(1)

                st.markdown("---")

                # Mostrar advertencias de valores no encontrados
                if warnings_list:
                    with st.expander(f"⚠️ {len(warnings_list)} advertencia(s) de encoding", expanded=False):
                        for w in warnings_list[:50]:
                            st.markdown(f"- {w}")
                        if len(warnings_list) > 50:
                            st.markdown(f"- ... y {len(warnings_list) - 50} mas")

                st.markdown(f"#### ✅ {len(df_result):,} registros procesados")
                st.dataframe(df_result, use_container_width=True)

                # Distribucion de severidades
                dist = df_result["SEVERIDAD_PREDICHA"].value_counts().reset_index()
                dist.columns = ["Severidad", "Cantidad"]
                dist["Porcentaje"] = (dist["Cantidad"] / len(df_result) * 100).round(1)

                fig_dist = px.bar(
                    dist,
                    x="Severidad",
                    y="Cantidad",
                    color="Severidad",
                    color_discrete_map={
                        "LEVE":   "#42A5F5",
                        "GRAVE":  "#FF7043",
                        "MORTAL": "#B71C1C",
                    },
                    title=f"Distribucion de Severidades Predichas — {modelo_lote}",
                    text=dist.apply(
                        lambda r: f"{int(r['Cantidad'])} ({r['Porcentaje']}%)", axis=1
                    ),
                )
                fig_dist.update_traces(textposition="outside")
                fig_dist.update_layout(height=430, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)

                # Botones de descarga
                st.markdown("#### Descargar resultados")
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.download_button(
                        label="⬇️ Descargar CSV",
                        data=df_result.to_csv(index=False).encode("utf-8"),
                        file_name="predicciones_incidentes.csv",
                        mime="text/csv",
                    )
                with dc2:
                    buf_excel = generar_excel_descarga(df_result)
                    st.download_button(
                        label="⬇️ Descargar Excel",
                        data=buf_excel.getvalue(),
                        file_name="predicciones_incidentes.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
