# NÍTIDO · App de Scoring — Instrucciones de despliegue

## Estructura de archivos necesaria

```
nitido_app/
├── app.py                ← Aplicación principal de Streamlit
├── requirements.txt      ← Dependencias de Python
├── exportar_modelo.py    ← Script para generar los .pkl desde el cuaderno
├── modelo_lr.pkl         ← ⚠️ Debes generarlo tú (ver Paso 1)
├── scaler.pkl            ← ⚠️ Debes generarlo tú (ver Paso 1)
└── umbral_final.pkl      ← ⚠️ Debes generarlo tú (ver Paso 1)
```

---

## Paso 1 — Generar los archivos del modelo (.pkl)

Los archivos `.pkl` contienen el modelo entrenado, el scaler y el umbral.
**No se incluyen en el repositorio** porque dependen de tu cuaderno.

### Opción A — Desde Google Colab

Al final de tu cuaderno (después de la Acción 12), agrega una celda nueva con:

```python
import joblib

# Guardar en Google Drive
joblib.dump(modelo_final,  '/content/drive/MyDrive/modelo_lr.pkl')
joblib.dump(scaler,        '/content/drive/MyDrive/scaler.pkl')
joblib.dump(UMBRAL_FINAL,  '/content/drive/MyDrive/umbral_final.pkl')
print("✅ Exportados.")
```

Luego descarga los 3 archivos desde Google Drive a tu computador.

### Opción B — Desde Jupyter local

Ejecuta `exportar_modelo.py` en la misma sesión donde entrenaste el modelo:

```bash
# Pega el contenido de exportar_modelo.py al final de tu notebook
# o ejecútalo como script si tienes las variables en el entorno
python exportar_modelo.py
```

---

## Paso 2 — Probar la app localmente (VS Code)

### 2.1 Crear el entorno virtual

```bash
# En la terminal de VS Code, desde la carpeta nitido_app/
python -m venv venv

# Activar el entorno:
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

### 2.2 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2.3 Verificar que tienes los 3 archivos .pkl

```bash
ls *.pkl
# Debes ver: modelo_lr.pkl  scaler.pkl  umbral_final.pkl
```

### 2.4 Lanzar la app

```bash
streamlit run app.py
```

La app se abre automáticamente en `http://localhost:8501`.
Si no, cópiala manualmente en tu navegador.

---

## Paso 3 — Subir a Streamlit Cloud

### 3.1 Crear repositorio en GitHub

1. Ve a [github.com/new](https://github.com/new) y crea un repositorio **privado** o público.
2. Sube la carpeta completa:

```bash
cd nitido_app/
git init
git add .
git commit -m "NITIDO app inicial"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/nitido-app.git
git push -u origin main
```

> ⚠️ Los archivos `.pkl` también deben subirse al repositorio.
> Si son grandes (>100 MB) usa Git LFS: `git lfs track "*.pkl"`.
> En este caso el modelo de regresión logística es muy pequeño (~KB), no hay problema.

### 3.2 Conectar con Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
2. Clic en **"New app"**.
3. Selecciona tu repositorio, rama `main` y archivo `app.py`.
4. Clic en **"Deploy"**.
5. Streamlit instalará las dependencias automáticamente desde `requirements.txt`.

### 3.3 Obtener el link público

Una vez desplegada, el link será algo como:
```
https://TU_USUARIO-nitido-app-app-XXXXX.streamlit.app
```

Copia ese link y pégalo en la **Acción 19** del cuaderno técnico.

---

## Paso 4 — Ajustes de diseño en app.py

Todo el diseño visual está centralizado en la sección **2. ESTILOS CSS** del archivo `app.py`.

| Qué cambiar | Dónde en app.py |
|-------------|-----------------|
| Colores principales | Variables `:root` en el bloque `<style>` |
| Fuentes | `@import url(...)` y `font-family` en los selectores |
| Textos descriptivos de variables | Diccionario `DESCRIPCIONES` (sección 4) |
| Variables top para contrafactual | Lista `TOP_VARS` (sección 4) |
| Métricas de referencia (AUC, F1…) | Últimos `st.metric()` en sección 8 |
| Umbral por defecto si falla el .pkl | No aplica; la app para con error claro |

---

## Preguntas frecuentes

**¿Qué pasa si el umbral_final.pkl tiene un valor diferente al del cuaderno?**
La app usa el valor exacto del archivo. Si re-entrenas el modelo, vuelve a exportar los 3 pkl.

**¿Se puede cambiar el modelo de LR a XGBoost?**
Sí, reemplaza `modelo_lr.pkl` con el XGBoost entrenado y asegúrate de que
`calcular_shap()` use `shap.TreeExplainer` en vez de `LinearExplainer`.

**¿Funciona sin internet?**
Sí, localmente. En Streamlit Cloud necesita internet para instalar paquetes la primera vez.
