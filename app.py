# =============================================================================
# NÍTIDO — App de Scoring de Candidatos  v2
# Reto Machine Learning II · Universidad Externado de Colombia
#
# Ejecutar localmente con: streamlit run app.py
#
# PALETA DE COLORES:
#   Cyan    → #6ed3cf   (acentos positivos, barras SHAP positivas)
#   Azul    → #1b3153   (encabezados, sidebar, elementos principales)
#   Blanco  → #ffffff   (fondo general)
#   Gris    → #f4f7fa   (fondo de tarjetas)
#
# CORRECCIONES v2:
#   - SHAP local se recalcula en cada evaluación (sin cache por candidato)
#   - Variables del sidebar agrupadas por tipo: Continuas / Ordinales / Binarias
#   - Colores de texto en contrafactual corregidos (ya no blanco sobre blanco)
#   - Nuevo tab: SHAP Global (importancia media sobre datos de referencia)
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend sin pantalla, necesario en Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE PÁGINA  (debe ser la PRIMERA llamada Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NÍTIDO · Scoring de Candidatos",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTILOS CSS
#    Todo el diseño visual está centralizado aquí.
#    Para cambiar colores edita las variables en :root.
#    Para cambiar fuentes edita el @import y los font-family.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Paleta principal ───────────────────────────────────────────────────── */
:root {
    --cyan:        #6ed3cf;
    --cyan-light:  #d0f2f1;
    --azul:        #1b3153;
    --azul-medio:  #254470;
    --azul-claro:  #e8eef6;
    --blanco:      #ffffff;
    --gris-fondo:  #f4f7fa;
    --gris-borde:  #dde3ec;
    --texto:       #1a1f2e;
    --texto-suave: #5a6478;
    --verde:       #1e7a4a;
    --verde-bg:    #e8faf4;
    --rojo:        #c0392b;
    --rojo-bg:     #fdf0ee;
}

/* ── Fondo y tipografía global ──────────────────────────────────────────── */
.stApp {
    background-color: var(--blanco);
    font-family: 'DM Sans', sans-serif;
    color: var(--texto);
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, var(--azul) 0%, var(--azul-medio) 100%);
}
/* Texto blanco en todo el sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #ffffff !important;
}
/* Etiquetas de sliders y selects en cyan */
[data-testid="stSidebar"] .stSlider    > label > div,
[data-testid="stSidebar"] .stSelectbox > label > div,
[data-testid="stSidebar"] .stRadio     > label > div {
    color: var(--cyan) !important;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(110,211,207,0.25);
    margin: 0.6rem 0;
}

/* ── Encabezado ─────────────────────────────────────────────────────────── */
.header-box {
    background: linear-gradient(130deg, var(--azul) 0%, var(--azul-medio) 55%, #2e6b9e 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    box-shadow: 0 4px 20px rgba(27,49,83,0.18);
}
.header-logo  { font-family:'DM Serif Display',serif; font-size:3rem; color:var(--cyan); line-height:1; }
.header-title { font-size:2.0rem; font-weight:1000; color:#fff; margin:0; }
.header-sub   { font-size:0.88rem; color:rgba(110,211,207,.9); margin:.2rem 0 0 0; }

/* ── Tarjetas de resultado ──────────────────────────────────────────────── */
.card-ok {
    background: var(--verde-bg);
    border: 2px solid #27ae60;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(39,174,96,.1);
}
.card-ko {
    background: var(--rojo-bg);
    border: 2px solid #e74c3c;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(231,76,60,.1);
}
.card-emoji  { font-size:2.6rem; }
.card-label  { font-size:1.5rem; font-weight:700; margin:.3rem 0 0 0; }
.card-label-ok  { color: var(--verde); }
.card-label-ko  { color: var(--rojo); }
.card-desc   { font-size:.9rem; color: var(--texto-suave); margin:.25rem 0 0 0; }

/* ── Métrica numérica ───────────────────────────────────────────────────── */
.metric-box {
    background: var(--gris-fondo);
    border: 1px solid var(--gris-borde);
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    height: 100%;
}
.metric-num   { font-family:'DM Serif Display',serif; font-size:3rem; color:var(--azul); line-height:1.1; }
.metric-label { font-size:.78rem; color:var(--texto-suave);letter-spacing:.08em; margin-top:.3rem; }

/* ── Título de sección ──────────────────────────────────────────────────── */
.sec-title {
    font-family:'DM Serif Display',serif;
    font-size:2.00rem;
    color: var(--azul);
    border-bottom: 2px solid var(--cyan);
    padding-bottom:.35rem;
    margin: 1.6rem 0 .9rem 0;
    display: inline-block;
}

/* ── Bloque de nota en sidebar ──────────────────────────────────────────── */
.sidebar-nota {
    background: rgba(110,211,207,.13);
    border-left: 3px solid var(--cyan);
    border-radius: 0 8px 8px 0;
    padding: .6rem .85rem;
    font-size: .8rem;
    color: rgba(255,255,255,.88) !important;
    margin: .4rem 0;
}

/* ── Badge umbral ───────────────────────────────────────────────────────── */
.badge-u {
    display:inline-block;
    background: var(--azul);
    color: var(--cyan);
    font-size:.78rem; font-weight:600;
    padding:.18rem .65rem;
    border-radius:20px; letter-spacing:.04em;
}

/* ── Tarjeta contrafactual ──────────────────────────────────────────────── */
/* IMPORTANTE: todos los textos dentro de .contra-* tienen color explícito
   para evitar que hereden el blanco del sidebar u otros contextos oscuros */
.contra-wrap {
    background: var(--azul-claro);
    border: 1px solid var(--gris-borde);
    border-left: 4px solid var(--cyan);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: .6rem;
}
.contra-head { font-size:.85rem; color: var(--texto-suave) !important; margin-bottom:.7rem; }
.contra-row  {
    display:flex; align-items:center; gap:.7rem;
    padding:.45rem 0; border-bottom:1px solid rgba(27,49,83,.08);
    font-size:.9rem;
}
.contra-row:last-child { border-bottom:none; }
.contra-var   { font-weight:700; color: var(--azul)  !important; min-width:42px; }
.contra-arrow { color: var(--cyan)  !important; font-size:1rem; }
.contra-delta-pos { color: var(--verde) !important; font-weight:600; }
.contra-delta-neg { color: var(--rojo)  !important; font-weight:600; }
.contra-note  { font-size:.78rem; color: var(--texto-suave) !important; margin-top:.6rem; }

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"]           { font-weight:600; color:var(--texto-suave); }
.stTabs [aria-selected="true"]         { color:var(--azul) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--cyan) !important; }

/* ── st.metric — forzar color oscuro (evita texto blanco en tema oscuro) ── */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] { color: var(--texto) !important; }
[data-testid="stMetricDelta"] { color: var(--verde) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARGA DE ARTEFACTOS  (cacheados para no recargar en cada interacción)
#
#    Archivos requeridos en la misma carpeta que app.py:
#      modelo_lr.pkl     → LogisticRegression entrenado
#      scaler.pkl        → StandardScaler ajustado sobre X_train
#      umbral_final.pkl  → float con el umbral óptimo (maximiza F1 en val)
#
#    Generarlos desde el cuaderno con: joblib.dump(objeto, "nombre.pkl")
#    Ver exportar_modelo.py para el bloque completo.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo NÍTIDO…")
def cargar_artefactos():
    """
    Carga todos los artefactos del modelo.

    Archivos requeridos en la misma carpeta que app.py:
        modelo_lr.pkl     → LogisticRegression entrenado
        scaler.pkl        → StandardScaler ajustado sobre X_train
        umbral_final.pkl  → float con el umbral optimo
        X_train_sc.npy    → array (N,18) con X_train ya escalado  ← CRITICO para SHAP

    Exportar desde el cuaderno con:
        import numpy as np, joblib
        joblib.dump(modelo_final,  "modelo_lr.pkl")
        joblib.dump(scaler,        "scaler.pkl")
        joblib.dump(UMBRAL_FINAL,  "umbral_final.pkl")
        np.save("X_train_sc.npy",  X_train_sc.values)
    """
    base   = os.path.dirname(os.path.abspath(__file__))
    modelo = joblib.load(os.path.join(base, "modelo_lr.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    umbral = joblib.load(os.path.join(base, "umbral_final.pkl"))

    # X_train_sc es OBLIGATORIO para SHAP correcto.
    # Sin el, el explainer usa solo al candidato como referencia
    # y los valores SHAP no cambian entre evaluaciones.
    X_train_sc_path = os.path.join(base, "X_train_sc.npy")
    if os.path.exists(X_train_sc_path):
        X_train_sc = np.load(X_train_sc_path)
    else:
        X_train_sc = None

    # Construir LinearExplainer UNA sola vez con X_train como referencia.
    # cache_resource garantiza que no se repite en cada interaccion.
    if X_train_sc is not None:
        explainer = shap.LinearExplainer(modelo, X_train_sc)
    else:
        explainer = None

    return modelo, scaler, float(umbral), explainer, X_train_sc

try:
    MODELO, SCALER, UMBRAL_FINAL, EXPLAINER, X_TRAIN_SC = cargar_artefactos()
    _artefactos_ok  = True
    _x_train_sc_ok  = X_TRAIN_SC is not None
except FileNotFoundError:
    _artefactos_ok  = False
    _x_train_sc_ok  = False


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSTANTES DEL PIPELINE
#    Deben coincidir exactamente con las del cuaderno técnico.
# ─────────────────────────────────────────────────────────────────────────────

# Orden exacto de columnas que espera el modelo (igual que X_train)
FEATURE_NAMES = [
    'x1','x2','x3','x4','x5',
    'x6','x7','x8','x9','x10',
    'x11','x12','x13','x14','x15',
    'x16','x17','x18'
]

# Clasificación de variables por tipo
# ─ Ajusta esta clasificación si el cuaderno la actualiza ─
VARS_CONTINUAS_PURAS = ['x1', 'x2', 'x10', 'x11', 'x14', 'x15']   # sliders con decimales
VARS_ORDINALES       = ['x3','x4','x6','x7','x8','x9','x12','x13','x16','x18']  # selectbox enteros
VARS_BINARIAS        = ['x5', 'x17']                                 # radio 0/1
# Las que van al scaler (todo excepto binarias, igual que el cuaderno)
VARS_ESCALAR         = [v for v in FEATURE_NAMES if v not in VARS_BINARIAS]

# Top 5 variables por Permutation Importance (Acción 14 del cuaderno)
TOP_VARS = ['x1', 'x2', 'x5', 'x3', 'x10']

# Rangos observados en el dataset (para sliders y validación)
RANGOS = {
    'x1': (0.0, 25.0), 'x2': (7.0, 100.0), 'x3': (1, 5),   'x4': (0, 8),
    'x5': (0, 1),      'x6': (22, 65),      'x7': (1, 6),   'x8': (1, 10),
    'x9': (0, 3),      'x10':(3.0, 5.0),    'x11':(0.1,100.0),'x12':(0,11),
    'x13':(0, 3),      'x14':(0.0, 24.0),   'x15':(1.0,50.0),'x16':(0,100),
    'x17':(0, 1),      'x18':(0, 10),
}

# Valores por defecto razonables para el formulario
DEFAULTS = {
    'x1':5.0,'x2':50.0,'x3':3,'x4':4,'x5':0,'x6':35,'x7':3,'x8':5,
    'x9':1,'x10':4.0,'x11':50.0,'x12':5,'x13':1,'x14':5.0,'x15':15.0,
    'x16':50,'x17':0,'x18':5,
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNCIONES DE PREPROCESAMIENTO Y MODELADO
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar(valores: dict) -> np.ndarray:
    """
    Arma el vector escalado de shape (1,18) a partir del dict de inputs.
    Replica el pipeline del cuaderno:
      - x5 y x17 (binarias) → sin escalar
      - resto               → scaler.transform()
    """
    df = pd.DataFrame([valores])[FEATURE_NAMES].astype(float)
    df[VARS_ESCALAR] = SCALER.transform(df[VARS_ESCALAR])
    return df.values  # (1, 18)


def predecir(x_sc: np.ndarray):
    """Retorna (predicción_binaria, probabilidad_clase_1)."""
    prob = MODELO.predict_proba(x_sc)[0, 1]
    pred = int(prob >= UMBRAL_FINAL)
    return pred, prob


def calcular_shap_local(x_sc: np.ndarray):
    """
    SHAP local para el candidato ingresado.

    Usa EXPLAINER (construido con X_train_sc en cargar_artefactos) como
    referencia fija. Esto garantiza que los valores SHAP cambien correctamente
    entre candidatos: cada valor refleja la desviacion del candidato
    respecto al valor base global del modelo (E[f(X)] sobre el entrenamiento).

    Si EXPLAINER es None (falta X_train_sc.npy) crea un explainer temporal
    usando solo al candidato — esto produce SHAP que NO varian entre evaluaciones.
    """
    if EXPLAINER is not None:
        # Caso correcto: explainer con X_train como referencia
        sv = EXPLAINER(x_sc)
    else:
        # Fallback degradado (sin X_train_sc.npy)
        exp_tmp = shap.LinearExplainer(MODELO, x_sc)
        sv = exp_tmp(x_sc)
    return sv.values[0], sv.base_values[0]


def interpretar_shap(sv_local, valores, top_n=5):
    """
    Genera explicaciones narrativas basadas en SHAP.
    """

    explicaciones_pos = []
    explicaciones_neg = []

    # Ordenar por impacto absoluto
    orden = np.argsort(np.abs(sv_local))[::-1]

    for idx in orden[:top_n]:

        var = FEATURE_NAMES[idx]
        shap_val = sv_local[idx]
        valor_real = valores[var]

        # Determinar intensidad
        intensidad = (
            "fuertemente" if abs(shap_val) > 0.5
            else "moderadamente" if abs(shap_val) > 0.2
            else "ligeramente"
        )

        # Variables binarias
        if var in VARS_BINARIAS:

            if shap_val > 0:
                texto = (
                    f"**{var}** favoreció {intensidad} la decisión "
                    f"porque tomó el valor `{int(valor_real)}`, "
                    f"lo que aumentó la probabilidad de aprobación."
                )
                explicaciones_pos.append(texto)

            else:
                texto = (
                    f"**{var}** redujo {intensidad} la probabilidad "
                    f"porque tomó el valor `{int(valor_real)}`, "
                    f"asociado con menor probabilidad de aprobación."
                )
                explicaciones_neg.append(texto)

        # Variables continuas / ordinales
        else:

            promedio = np.mean(X_TRAIN_SC[:, idx]) if X_TRAIN_SC is not None else 0

            posicion = (
                "alto" if x_sc[0][idx] > promedio
                else "bajo"
            )

            if shap_val > 0:
                texto = (
                    f"**{var}** tuvo un impacto positivo {intensidad} "
                    f"porque el valor ingresado (`{valor_real}`) es relativamente "
                    f"{posicion}, aumentando la probabilidad de aprobación."
                )
                explicaciones_pos.append(texto)

            else:
                texto = (
                    f"**{var}** disminuyó {intensidad} la probabilidad "
                    f"porque el valor ingresado (`{valor_real}`) es relativamente "
                    f"{posicion}, reduciendo la probabilidad de aprobación."
                )
                explicaciones_neg.append(texto)

    return explicaciones_pos, explicaciones_neg

def calcular_shap_global():
    """
    Importancia global: media del |SHAP| sobre todo X_train_sc.
    Usa EXPLAINER y X_TRAIN_SC precargados (correctos y eficientes).
    Si no hay X_train_sc.npy devuelve los coeficientes del modelo como proxy.
    """
    if EXPLAINER is not None and X_TRAIN_SC is not None:
        sv = EXPLAINER(X_TRAIN_SC)
        return np.abs(sv.values).mean(axis=0)
    else:
        # Proxy: valor absoluto de coeficientes de la regresion logistica
        return np.abs(MODELO.coef_[0])


def generar_contrafactual(x_sc: np.ndarray, paso=0.05, max_iter=300) -> dict:
    """
    Contrafactual greedy (replica Acción 17 del cuaderno).
    Itera sobre TOP_VARS: en cada paso aplica el cambio que más
    aumenta la probabilidad hasta superar UMBRAL_FINAL.
    Retorna dict {variable: delta_en_espacio_escalado}.
    """
    caso    = x_sc[0].copy()
    cambios = {}

    for _ in range(max_iter):
        prob = MODELO.predict_proba(caso.reshape(1,-1))[0,1]
        if prob >= UMBRAL_FINAL:
            break

        mejor_delta, mejor_var, mejor_dir = 0.0, None, 0.0

        for var in TOP_VARS:
            idx = FEATURE_NAMES.index(var)

            if var in VARS_BINARIAS:
                # Probar flip 0↔1
                val_flip        = 1.0 - caso[idx]
                tmp             = caso.copy()
                tmp[idx]        = val_flip
                d               = MODELO.predict_proba(tmp.reshape(1,-1))[0,1] - prob
                if d > mejor_delta:
                    mejor_delta, mejor_var, mejor_dir = d, var, val_flip - caso[idx]
            else:
                # Probar incremento y decremento
                for sign in [+paso, -paso]:
                    tmp      = caso.copy()
                    tmp[idx] += sign
                    d        = MODELO.predict_proba(tmp.reshape(1,-1))[0,1] - prob
                    if d > mejor_delta:
                        mejor_delta, mejor_var, mejor_dir = d, var, sign

        if mejor_var is None:
            break

        idx_m = FEATURE_NAMES.index(mejor_var)
        if mejor_var in VARS_BINARIAS:
            caso[idx_m]          = caso[idx_m] + mejor_dir
        else:
            caso[idx_m]         += mejor_dir
        cambios[mejor_var]       = cambios.get(mejor_var, 0) + mejor_dir

    return cambios


# ─────────────────────────────────────────────────────────────────────────────
# 6. FUNCIONES DE GRÁFICOS
#    Cada función devuelve una Figure de matplotlib cerrada con plt.close()
#    para evitar que Streamlit muestre figuras residuales de ejecuciones
#    anteriores (esa era la razón por la que el gráfico no se actualizaba).
# ─────────────────────────────────────────────────────────────────────────────

def _estilo_base(fig, ax):
    """Aplica fondo y grilla comunes a todos los gráficos."""
    fig.patch.set_facecolor('#f4f7fa')
    ax.set_facecolor('#f4f7fa')
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(colors='#5a6478', labelsize=9)


def figura_shap_local(shap_vals: np.ndarray, proba: float) -> plt.Figure:
    """
    Gráfico de barras horizontal (waterfall simplificado) para SHAP local.
    Muestra las 10 variables con mayor impacto absoluto.

    CORRECCIÓN v2: la figura se crea desde cero con plt.figure() y se cierra
    antes de retornar; esto fuerza a Streamlit a renderizar la nueva figura
    en lugar de reutilizar la del candidato anterior.
    """
    # Seleccionar top-10 por magnitud
    orden   = np.argsort(np.abs(shap_vals))[::-1][:10]
    nombres = [FEATURE_NAMES[i] for i in orden]
    vals    = [shap_vals[i]     for i in orden]
    colores = ['#6ed3cf' if v >= 0 else '#e74c3c' for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    _estilo_base(fig, ax)

    bars = ax.barh(
        nombres[::-1], vals[::-1],
        color=colores[::-1],
        edgecolor='white', linewidth=0.7, height=0.58
    )
    ax.axvline(0, color='#1b3153', lw=1.1, ls='--', alpha=0.45)
    ax.set_xlabel("Valor SHAP (contribución al log-odds)", fontsize=9, color='#5a6478')
    ax.set_title(
        f"Explicación local  ·  Probabilidad predicha: {proba:.1%}",
        fontsize=10, fontweight='bold', color='#1b3153', pad=10
    )

    # Etiquetas numéricas correctamente alineadas
    for bar, v in zip(bars, vals[::-1]):

        # Offset dinámico según tamaño del eje
        offset = max(abs(v) * 0.0001, 0.01)

        if v >= 0:
            x_text = v + offset
            ha = 'left'
        else:
            x_text = v - offset
            ha = 'right'

        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}",
            va='center',
            ha=ha,
            fontsize=8,
            color='#1b3153',
            fontweight='600'
        )

    # Leyenda
    #ax.legend(handles=[
    #    mpatches.Patch(color='#6ed3cf', label='↑ Empuja a aprobación'),
    #   mpatches.Patch(color='#e74c3c', label='↓ Empuja a rechazo'),
    #], fontsize=8, loc='lower right', framealpha=0.85)

    plt.tight_layout()
    return fig


def figura_shap_global(mean_shap: np.ndarray) -> plt.Figure:
    """
    Gráfico de importancia global: media del |SHAP| por variable.
    Muestra todas las variables ordenadas de mayor a menor importancia.
    """
    orden   = np.argsort(mean_shap)          # ascendente para barh
    nombres = [FEATURE_NAMES[i] for i in orden]
    vals    = [mean_shap[i]     for i in orden]

    # Degradado de color por importancia (más importante = más oscuro)
    max_v   = max(vals) if max(vals) > 0 else 1
    colores = [
        (
            int(110 - 50 * (v/max_v)),   # R: 110 → 60
            int(211 - 80 * (v/max_v)),   # G: 211 → 131
            int(207 - 60 * (v/max_v)),   # B: 207 → 147
        )
        for v in vals
    ]
    hex_cols = ['#%02x%02x%02x' % c for c in colores]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _estilo_base(fig, ax)

    bars = ax.barh(nombres, vals, color=hex_cols, edgecolor='white', linewidth=0.6, height=0.65)

    # Línea de referencia en la media
    media = np.mean(vals)
    ax.axvline(media, color='#1b3153', lw=1.0, ls=':', alpha=0.55, label=f'Media = {media:.4f}')

    # Etiquetas
    for bar, v in zip(bars, vals):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va='center', ha='left', fontsize=8, color='#1b3153')

    ax.set_xlabel("Media de |SHAP| — importancia global", fontsize=9, color='#5a6478')
    ax.set_title("Importancia global de variables (SHAP)",
                 fontsize=10, fontweight='bold', color='#1b3153', pad=10)
    ax.legend(fontsize=8, framealpha=0.8)

    plt.tight_layout()
    # NO cerrar aquí — se cierra después de st.pyplot()
    return fig


def figura_gauge(proba: float) -> plt.Figure:
    """
    Mini-gauge semicircular de probabilidad.
    Verde: prob alta · Naranja: media · Rojo: baja
    """
    fig, ax = plt.subplots(figsize=(3.2, 1.9), subplot_kw={'aspect':'equal'})
    fig.patch.set_facecolor('#f4f7fa')
    ax.set_facecolor('#f4f7fa')

    # Arco de fondo
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=14, color='#dde3ec', solid_capstyle='round')

    # Arco de progreso
    color_g = '#27ae60' if proba >= 0.65 else ('#e67e22' if proba >= 0.4 else '#e74c3c')
    theta_p = np.linspace(np.pi, np.pi - np.pi * proba, 200)
    ax.plot(np.cos(theta_p), np.sin(theta_p), lw=14, color=color_g, solid_capstyle='round')

    # Texto central
    ax.text(0, -0.15, f"{proba:.1%}", ha='center', va='center',
            fontsize=18, fontweight='bold', color='#1b3153',
            fontfamily='DejaVu Serif')
    ax.text(0, -0.48, "Probabilidad", ha='center', fontsize=7.5, color='#5a6478')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.7, 1.2)
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    # NO cerrar aquí — se cierra después de st.pyplot()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENCABEZADO con logo real
# ─────────────────────────────────────────────────────────────────────────────
import base64

def _img_b64(path: str) -> str:
    """Convierte una imagen a base64 para incrustarla en HTML."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Ruta del logo (mismo directorio que app.py)
_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_nitido1.png")
_logo_ok   = os.path.exists(_logo_path)

if _logo_ok:
    _logo_b64 = _img_b64(_logo_path)
    st.markdown(f"""
    <div class="header-box">
      <img src="data:image/png;base64,{_logo_b64}"
        style="height:200px;width:auto;object-fit:contain;opacity:.95;">
      <div style="border-left:2px solid rgba(110,211,207,.35);padding-left:1.4rem">
        <header class="header-title">Sistema de Pre-filtrado de Candidatos</header>
        <p class="header-sub">Regresión Logística &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; Contrafactuales accionables</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback si no se encuentra el logo
    st.markdown("""
    <div class="header-box">
      <div class="header-logo">N</div>
      <div>
        <p class="header-title">NÍTIDO · Sistema de Pre-filtrado de Candidatos</p>
        <p class="header-sub">Regresión Logística &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; Contrafactuales accionables</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

if not _artefactos_ok:
    st.error(
        "⚠️ Faltan archivos del modelo: `modelo_lr.pkl`, `scaler.pkl`, "
        "`umbral_final.pkl`. Generalos desde el cuaderno y colocalos "
        "en la misma carpeta que `app.py`. Ver `exportar_modelo.py`."
    )
    st.stop()

# Aviso si falta X_train_sc.npy (SHAP funcionara pero con valores fijos)
if not _x_train_sc_ok:
    st.warning(
        "⚠️ Falta `X_train_sc.npy` — el grafico SHAP no variara entre candidatos. "
        "Exportalo desde el cuaderno con: `np.save('X_train_sc.npy', X_train_sc.values)` "
        "y coloca el archivo junto a `app.py`."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. SIDEBAR — INPUTS ORGANIZADOS POR TIPO DE VARIABLE
#
#    GRUPO A — Variables continuas  (sliders con decimales): x1, x2, x10, x11, x14, x15
#    GRUPO B — Variables ordinales/enteras (selectbox):      x3,x4,x6,x7,x8,x9,x12,x13,x16,x18
#    GRUPO C — Variables binarias (radio 0/1):               x5, x17
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo en el sidebar
    if _logo_ok:
        st.markdown(f"""
        <div style='text-align:center;padding:1rem 0 1.2rem'>
         
          <p style='font-size:1.72rem;color:rgba(255,255,255,.5);margin:.4rem 0 0 0'>
            Datos del candidato
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 1.2rem'>
          <span style='font-family:"DM Serif Display",serif;font-size:2rem;color:#6ed3cf'>NÍTIDO</span>
          <p style='font-size:.72rem;color:rgba(255,255,255,.55);margin:0'>Datos del candidato</p>
        </div>
        """, unsafe_allow_html=True)

    # ── GRUPO A: Variables continuas ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='color:#6ed3cf;font-size:.78rem;font-weight:700;"
        "letter-spacing:.08em;text-transform:uppercase;margin-bottom:.3rem'>"
        "Variables continuas</p>", unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-nota">Valores numéricos con decimales. Usa el slider o escribe el número.</div>',
                unsafe_allow_html=True)

    x1  = st.slider("x1  · Continua  [0 – 25]",   0.0,  25.0, float(DEFAULTS['x1']),  step=0.1)
    x2  = st.slider("x2  · Continua  [7 – 100]",   7.0, 100.0, float(DEFAULTS['x2']),  step=0.5)
    x10 = st.slider("x10 · Continua  [3 – 5]",     3.0,   5.0, float(DEFAULTS['x10']), step=0.01)
    x11 = st.slider("x11 · Continua  [0.1 – 100]", 0.1, 100.0, float(DEFAULTS['x11']), step=0.1)
    x14 = st.slider("x14 · Continua  [0 – 24]",    0.0,  24.0, float(DEFAULTS['x14']), step=0.1)
    x15 = st.slider("x15 · Continua  [1 – 50]",    1.0,  50.0, float(DEFAULTS['x15']), step=0.1)

    # ── GRUPO B: Variables ordinales/enteras ──────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='color:#6ed3cf;font-size:.78rem;font-weight:700;"
        "letter-spacing:.08em;text-transform:uppercase;margin-bottom:.3rem'>"
        "Variables ordinales / enteras</p>", unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-nota">Valores enteros con categorías ordenadas. Selecciona de la lista.</div>',
                unsafe_allow_html=True)

    # Usar 2 columnas para compactar
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        x3  = st.selectbox("x3  [1–5]",   [1,2,3,4,5],          index=DEFAULTS['x3']-1)
        x7  = st.selectbox("x7  [1–6]",   [1,2,3,4,5,6],        index=DEFAULTS['x7']-1)
        x9  = st.selectbox("x9  [0–3]",   [0,1,2,3],            index=DEFAULTS['x9'])
        x13 = st.selectbox("x13 [0–3]",   [0,1,2,3],            index=DEFAULTS['x13'])
        x18 = st.selectbox("x18 [0–10]",  list(range(11)),       index=DEFAULTS['x18'])
    with col_s2:
        x4  = st.selectbox("x4  [0–8]",   list(range(9)),        index=DEFAULTS['x4'])
        x8  = st.selectbox("x8  [1–10]",  list(range(1,11)),     index=DEFAULTS['x8']-1)
        x12 = st.selectbox("x12 [0–11]",  list(range(12)),       index=DEFAULTS['x12'])
        x16 = st.selectbox("x16 [0–100]", list(range(0,101,5)),  index=list(range(0,101,5)).index(DEFAULTS['x16']))
        x6  = st.selectbox("x6  [22–65]", list(range(22,66)),    index=DEFAULTS['x6']-22)

    # ── GRUPO C: Variables binarias ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='color:#6ed3cf;font-size:.78rem;font-weight:700;"
        "letter-spacing:.08em;text-transform:uppercase;margin-bottom:.3rem'>"
        "Variables binarias</p>", unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-nota">⚠️ x5 tiene Disparate Impact Ratio = 1.47 (fuera de [0.80, 1.25]). '
        'Ver pestaña Auditoría.</div>',
        unsafe_allow_html=True
    )
    x5  = st.radio("x5  [0 / 1]",  [0, 1], index=DEFAULTS['x5'],  horizontal=True)
    x17 = st.radio("x17 [0 / 1]", [0, 1], index=DEFAULTS['x17'], horizontal=True)

    st.markdown("---")
    evaluar = st.button("Evaluar candidato", use_container_width=True, type="primary")
    st.markdown(
        f'<p style="text-align:center;margin-top:.7rem;font-size:.8rem;">'
        f'Umbral activo: <span class="badge-u">{UMBRAL_FINAL:.3f}</span></p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. LÓGICA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if evaluar:

    # 9.1 — Armar dict de valores (conversión a float garantizada)
    valores = {
        'x1':float(x1),'x2':float(x2),'x3':float(x3),'x4':float(x4),
        'x5':float(x5),'x6':float(x6),'x7':float(x7),'x8':float(x8),
        'x9':float(x9),'x10':float(x10),'x11':float(x11),'x12':float(x12),
        'x13':float(x13),'x14':float(x14),'x15':float(x15),'x16':float(x16),
        'x17':float(x17),'x18':float(x18),
    }

    # 9.2 — Pipeline completo
    x_sc        = preprocesar(valores)
    pred, proba = predecir(x_sc)

    # ── Fila de resultado ─────────────────────────────────────────────────
    col_res, col_gauge, col_nivel = st.columns([2.2, 1.2, 1])

    with col_res:
        if pred == 1:
            st.markdown("""
            <div class="card-ok">
              <div class="card-emoji">✅</div>
              <p class="card-label card-label-ok">Candidato Aprobado</p>
              <p class="card-desc">Avanza a la siguiente etapa del proceso.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card-ko">
              <div class="card-emoji">❌</div>
              <p class="card-label card-label-ko">Candidato Descartado</p>
              <p class="card-desc">No superó el umbral de selección.</p>
            </div>""", unsafe_allow_html=True)

    with col_gauge:
        # Gauge semicircular — se genera NUEVO en cada evaluación
        fig_g = figura_gauge(proba)
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

    with col_nivel:
        nivel      = "Alto"    if proba >= 0.65 else ("Medio" if proba >= 0.4 else "Bajo")
        col_nivel_c = "#1e7a4a" if proba >= 0.65 else ("#d68910" if proba >= 0.4 else "#c0392b")
        st.markdown(f"""
        <div class="metric-box" style="margin-top:.3rem">
          <div class="metric-num" style="color:{col_nivel_c};font-size:2.2rem">{nivel}</div>
          <div class="metric-label">Probabilidad de avanzar</div>
          <div style="font-size:.78rem;color:#5a6478;margin-top:.4rem">
            Umbral: {UMBRAL_FINAL:.3f}
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Tabs de análisis ──────────────────────────────────────────────────
    st.markdown('<header class="sec-title">Análisis detallado</header>', unsafe_allow_html=True)

    tab_local, tab_contra, tab_datos, tab_global = st.tabs([                         #tab_global            
        "¿Qué influyó en el resultado?", "Detalles o recomendaciones para aprobar", "Datos del candidato", "Variables más importantes"
    ])

    # ── Tab 1: SHAP Local ─────────────────────────────────────────────────
    with tab_local:
        st.markdown(
            "**¿Por qué el modelo tomó esta decisión para este candidato?**  "
            "Las barras <span style='color:#6ed3cf;font-weight:700'>cyan</span> "
            "aumentan la probabilidad de aprobación; las "
            "<span style='color:#e74c3c;font-weight:700'>rojas</span> la reducen.",
            unsafe_allow_html=True
        )

        # SHAP local: sin cache → siempre recalcula para el candidato actual
        with st.spinner("Calculando SHAP local…"):
            sv_local, base_val = calcular_shap_local(x_sc)

        # plt.clf() limpia el estado global de matplotlib antes de crear la figura
        plt.clf()
        fig_local = figura_shap_local(sv_local, proba)
        st.pyplot(fig_local, use_container_width=True)
        plt.close(fig_local)   # cierra DESPUÉS de renderizar

        with st.expander("Ver tabla completa de valores SHAP"):
            df_sv = pd.DataFrame({
                'Variable':        FEATURE_NAMES,
                'Valor ingresado': [valores[v] for v in FEATURE_NAMES],
                'SHAP':            sv_local,
            }).sort_values('SHAP', key=abs, ascending=False)
            df_sv['SHAP'] = df_sv['SHAP'].map('{:+.4f}'.format)
            st.dataframe(df_sv, use_container_width=True, hide_index=True)

    # ── Tab 2: SHAP Global ────────────────────────────────────────────────
    with tab_global:
        if _x_train_sc_ok:
            st.markdown(
                "**Importancia global de variables** "  # media del |SHAP| calculada sobre el conjunto de entrenamiento completo (X_train_sc.npy)

            )
        else:
            st.warning(
                "Sin `X_train_sc.npy` se muestran los coeficientes del modelo como proxy. "
                "Exporta ese archivo desde el cuaderno para ver la importancia SHAP real."
            )
            st.markdown("**Importancia global (proxy por coeficientes del modelo):**")

        # Usamos la muestra del candidato como referencia (1 punto)
        # Si tienes X_train_sc.npy exportado del cuaderno, cámbialo por:
        # X_ref = np.load("X_train_sc.npy")
        # y el gráfico será más representativo.
        with st.spinner("Calculando SHAP global…"):
            mean_sv = calcular_shap_global()   # usa X_train_sc precargado

        plt.clf()
        fig_glob = figura_shap_global(mean_sv)
        st.pyplot(fig_glob, use_container_width=True)
        plt.close(fig_glob)

        st.caption(
            "ℹ️ Interpretación: mayor barra = la variable tiene más influencia promedio "
            "sobre las predicciones del modelo."
        )

        # Tabla de importancias globales
        with st.expander("Ver tabla de importancias globales"):
            df_glob = pd.DataFrame({
                'Variable':    FEATURE_NAMES,
                'Importancia': mean_sv,
            }).sort_values('Importancia', ascending=False)
            df_glob['Importancia'] = df_glob['Importancia'].map('{:.5f}'.format)
            st.dataframe(df_glob, use_container_width=True, hide_index=True)

    # ── Tab 3: Contrafactual ──────────────────────────────────────────────
    with tab_contra:
        exp_pos, exp_neg = interpretar_shap(sv_local, valores)
        if pred == 1:
            st.success("✅ El candidato fue aprobado.")
            st.markdown("### Variables que más impulsaron la aprobación")
            for e in exp_pos[:5]:
                st.markdown(f"- {e}")

        else:
            st.markdown(
                "### **¿Qué tendría que cambiar este candidato para ser aprobado?**  \n"
                "El algoritmo busca los cambios en las variables más influyentes "
                f"(Top 5: {', '.join(TOP_VARS)}) que llevarían la probabilidad por encima "
                f"del umbral **{UMBRAL_FINAL:.3f}**."
            )
            

            with st.spinner("Generando contrafactual…"):
                cambios = generar_contrafactual(x_sc)
                
            if not cambios:
                st.warning(
                    "No se encontró un contrafactual alcanzable solo con las variables top. "
                    "El perfil del candidato está muy alejado del umbral."
                )
            else:
                # Calcular probabilidad contrafactual
                x_cf = x_sc.copy()
                for var, delta in cambios.items():
                    x_cf[0, FEATURE_NAMES.index(var)] += delta
                proba_cf = MODELO.predict_proba(x_cf)[0, 1]

                # Métricas — usar st.metric con colores corregidos vía CSS
                c1, c2, c3 = st.columns(3)
                c1.metric("Prob. actual",         f"{proba:.1%}")
                c2.metric("Prob. contrafactual",  f"{proba_cf:.1%}",
                          delta=f"+{(proba_cf - proba):.1%}")
                c3.metric("Cambios aplicados",    str(len(cambios)))

                # Tarjeta de cambios con colores explícitos (no heredan blanco)
                st.markdown('<div class="contra-wrap">', unsafe_allow_html=True)
                st.markdown( 
                    ' #### Cambios mínimos sugeridos ',
                    unsafe_allow_html=True
                )

                for var, delta_std in cambios.items():

                    idx = FEATURE_NAMES.index(var)

                    valor_actual = valores[var]

                    # Variables binarias
                    if var in VARS_BINARIAS:

                        valor_nuevo = int(1 - valor_actual)

                        signo = "Subir ▲" if valor_nuevo > valor_actual else "Bajar ▼"  
                        clase = "contra-delta-pos" if valor_nuevo > valor_actual else "contra-delta-neg"

                        texto_delta = f"{int(valor_actual)} → {valor_nuevo}"

                    else:

                        # Buscar desviación estándar correcta del scaler
                        idx_scaler = VARS_ESCALAR.index(var)
                        std_real = SCALER.scale_[idx_scaler]

                        # Convertir delta estandarizado → escala original
                        delta_real = delta_std * std_real

                        valor_nuevo = valor_actual + delta_real

                        # Limitar a rango permitido
                        min_v, max_v = RANGOS[var]
                        valor_nuevo = np.clip(valor_nuevo, min_v, max_v)

                        signo = "Subir ▲" if delta_real > 0 else "Bajar ▼"
                        clase = "contra-delta-pos" if delta_real > 0 else "contra-delta-neg"

                        texto_delta = (
                            f"{valor_actual:.2f} → {valor_nuevo:.2f} "
                            f"({delta_real:+.2f})"
                        )

                    st.markdown(
                        f'<div class="contra-row">'
                        f'<span class="contra-var">{var}</span>'
                        f'<span class="contra-arrow">{signo}</span>'
                        f'<span class="{clase}">{texto_delta}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<p class="contra-note">Se muestran cambios aproximados en la escala '
                    'original de cada variable.</p>',
                    unsafe_allow_html=True
                )
                
                st.markdown("### Variables que más redujeron la probabilidad")

                for e in exp_neg[:5]:
                    st.markdown(f"- {e}")

                st.markdown('</div>', unsafe_allow_html=True)
                
                

    # ── Tab 4: Datos ingresados ───────────────────────────────────────────
    with tab_datos:
        st.markdown("**Valores del formulario para este candidato:**")
        tipo_map = {v: ('Binaria' if v in VARS_BINARIAS
                        else ('Continua' if v in VARS_CONTINUAS_PURAS else 'Ordinal/Entera'))
                    for v in FEATURE_NAMES}
        df_in = pd.DataFrame({
            'Variable': FEATURE_NAMES,
            'Tipo':     [tipo_map[v] for v in FEATURE_NAMES],
            'Valor':    [valores[v]  for v in FEATURE_NAMES],
        })
        st.dataframe(df_in, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PANTALLA DE BIENVENIDA (cuando aún no se presionó "Evaluar")
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style='background:#f4f7fa;border:1px dashed #dde3ec;border-radius:14px;
                padding:3rem;text-align:center;margin-top:.5rem'>
      <div style='font-size:3rem;margin-bottom:.8rem'>👤</div>
      <p style='font-size:2.0rem;font-weight:600;color:#1b3153;margin-bottom:.5rem'>
        Ingresa los datos del candidato en el panel izquierdo
      </p>
      <p style='color:#5a6478;max-width:520px;margin:0 auto;font-size:.9rem'>
        Completa las variables por grupo (continuas, ordinales, binarias)
        y presiona <strong>Evaluar candidato</strong>. Obtendrás la predicción,
        el gráfico SHAP local, la importancia global y el contrafactual si aplica.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Métricas de referencia del modelo (del cuaderno, Acción 13)
    # Actualiza estos valores si re-entrenas el modelo
    st.markdown('<h3 class="sec-title">Métricas del modelo en test</h3>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC",  "~0.82",            help="Capacidad discriminativa global")
    m2.metric("F1-Score", "~0.78",            help="Balance Precision/Recall con umbral óptimo")
    m3.metric("Recall",   "~0.80",            help="Candidatos válidos correctamente identificados")
    m4.metric("Umbral",   f"{UMBRAL_FINAL:.3f}", help="Maximiza F1 sobre el conjunto de validación")
