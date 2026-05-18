# NÍTIDO · Sistema de Pre-filtrado de Candidatos

**Universidad Externado de Colombia · Pregrado en Ciencia de Datos**  
**Curso:** Machine Learning II · **Docente:** Alber Montenegro  
**Equipo:** Cerinza · González

---

## ¿Qué es esta aplicación?

NÍTIDO es un sistema de inteligencia artificial que analiza el perfil de un candidato y predice si debería avanzar a la etapa de entrevistas. Fue construido para apoyar el proceso de pre-filtrado de hojas de vida en empresas que reciben grandes volúmenes de aplicaciones.

La aplicación no reemplaza la decisión humana. Funciona como un primer filtro que le presenta al evaluador tres cosas:

1. **Una predicción** — si el candidato avanza o no, con la probabilidad asociada.
2. **Una explicación** — qué variables del perfil influyeron más en esa decisión y en qué dirección.
3. **Un camino de mejora** — si el candidato fue rechazado, qué tendría que cambiar para ser aprobado.

---

## ¿Cómo se usa?

Ingresa a la app en el siguiente link:

> 🔗 **[[[https://nitidoapp-ejyamcrur2mnnhvddmctm4.streamlit.app/](https://nitidoapp-ejyamcrur2mnnhvddmctm4.streamlit.app/)]]**

En el panel izquierdo encontrarás las 18 variables del perfil del candidato, organizadas en tres grupos:

- **Variables continuas** — valores numéricos con decimales (experiencia, puntajes, etc.)
- **Variables ordinales** — escalas enteras con categorías ordenadas (niveles, rangos, etc.)
- **Variables binarias** — características de sí o no (0 o 1)

Ajusta los valores y presiona **"Evaluar candidato"**. La app muestra el resultado en segundos.

---

## ¿Qué muestra la app?

### Resultado principal
Una tarjeta verde (aprobado) o roja (rechazado) con la probabilidad exacta y el nivel de confianza del modelo.

### Pestaña SHAP Local
Explica la decisión para ese candidato específico. Cada barra representa una variable:
- **Barra cyan** → esa variable empujó al candidato hacia la aprobación
- **Barra roja** → esa variable lo empujó hacia el rechazo

Debajo del gráfico aparece un resumen en lenguaje claro: qué variables lo aprobaron, qué variables lo hundieron, y cuáles debería mejorar.

### Pestaña SHAP Global
Muestra qué variables son más importantes para el modelo en general (no solo para un candidato). Incluye un segundo gráfico que responde: *¿subir el valor de esta variable ayuda o perjudica las chances de aprobación?*

### Pestaña Contrafactual
Si el candidato fue rechazado, muestra el camino mínimo para ser aprobado: qué variables cambiar y en qué dirección.

---

## El modelo

El sistema usa una **Regresión Logística** entrenada sobre 5.000 candidatos históricos de NÍTIDO. Se eligió este modelo sobre alternativas más complejas (Random Forest, XGBoost) por una razón regulatoria: la nueva Ley Estatutaria de Inteligencia Artificial en Colombia exige que los sistemas automatizados que toman decisiones sobre personas sean explicables ante el Ministerio. La Regresión Logística es explicable por construcción — cada coeficiente tiene un significado directo y auditable.

| Métrica | Valor en test |
|---------|--------------|
| AUC-ROC | ~0.82 |
| F1-Score | ~0.78 |
| Recall | ~0.80 |
| Umbral de decisión | Ver app |

El modelo no presenta sobreajuste significativo (diferencia train–test menor a 0.05 en todas las métricas).

---

## Hallazgo crítico: sesgo en x5

El análisis de equidad detectó que la variable **x5** genera un Disparate Impact Ratio de **1.47**, fuera del rango aceptable de [0.80 – 1.25]. Esto significa que el grupo con x5 = 1 es aprobado a una tasa casi 47% mayor que el grupo con x5 = 0.

Mientras no se confirme qué representa x5, el modelo **no debería desplegarse en producción** sin antes identificar si esta variable corresponde a un atributo protegido. Ver Acción 18 del cuaderno técnico para las recomendaciones completas.

---

## Estructura del repositorio

```
nitido_app/
├── app.py              → Aplicación principal
├── requirements.txt    → Dependencias de Python
├── modelo_lr.pkl       → Modelo de Regresión Logística entrenado
├── scaler.pkl          → Normalizador ajustado sobre datos de entrenamiento
├── umbral_final.pkl    → Umbral óptimo de decisión (maximiza F1)
├── X_train_sc.npy      → Datos de entrenamiento escalados (necesarios para SHAP)
└── logo_nitido1.png     → Logo de la empresa
```

---



---

*Construido con Python · scikit-learn · SHAP · Streamlit*  
*Datos anonimizados por acuerdo de confidencialidad con los clientes de NÍTIDO*
