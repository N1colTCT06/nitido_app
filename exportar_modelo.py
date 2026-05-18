# =============================================================================
# exportar_modelo.py
#
# Ejecuta este script UNA VEZ desde tu cuaderno de Colab (o Jupyter)
# DESPUÉS de haber entrenado el modelo y calculado el umbral.
#
# Copia las últimas celdas de tu .ipynb que tengan:
#   - modelo_final  (LogisticRegression entrenado)
#   - scaler        (StandardScaler ajustado sobre train)
#   - UMBRAL_FINAL  (float, umbral óptimo de F1)
#
# Luego ejecuta: python exportar_modelo.py
# O pega este bloque al final de tu cuaderno como celda de código.
# =============================================================================

import joblib
import os

# ── Carpeta de destino ──────────────────────────────────────────────────────
# Cambia esta ruta a la carpeta donde tienes app.py
CARPETA_APP = "./"   # si exportas desde dentro de la carpeta nitido_app

os.makedirs(CARPETA_APP, exist_ok=True)

# ── Exportar ────────────────────────────────────────────────────────────────
# Estas variables deben estar definidas en tu sesión de Python/Colab:
#   modelo_final  → LogisticRegression (ya entrenado y ajustado)
#   scaler        → StandardScaler (ya ajustado sobre X_train)
#   UMBRAL_FINAL  → float (por ejemplo 0.4823)

joblib.dump(modelo_final,  os.path.join(CARPETA_APP, "modelo_lr.pkl"))
joblib.dump(scaler,        os.path.join(CARPETA_APP, "scaler.pkl"))
joblib.dump(UMBRAL_FINAL,  os.path.join(CARPETA_APP, "umbral_final.pkl"))

print("✅ Artefactos exportados:")
print(f"   modelo_lr.pkl    → {os.path.join(CARPETA_APP, 'modelo_lr.pkl')}")
print(f"   scaler.pkl       → {os.path.join(CARPETA_APP, 'scaler.pkl')}")
print(f"   umbral_final.pkl → {os.path.join(CARPETA_APP, 'umbral_final.pkl')}")
print("\nCopia estos 3 archivos a la carpeta de tu app de Streamlit.")
