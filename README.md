# 🎓 Laboratorio de Riesgos – Universidad del Azuay

Aplicación web interactiva desarrollada con Streamlit para el aprendizaje didáctico sobre gestión de riesgos financieros.

## 📋 Descripción

Esta aplicación educativa tipo juego está diseñada para que estudiantes aprendan sobre tres tipos principales de riesgo:

1. **Riesgo de Mercado**: Comprende la volatilidad de activos financieros y el impacto de shocks de mercado
2. **Riesgo Financiero**: Utiliza el modelo Altman Z'-Score para evaluar el riesgo de quiebra empresarial
3. **Riesgo Macroeconómico**: Analiza cómo las variables macroeconómicas afectan las decisiones financieras

## 🚀 Instalación

### Requisitos previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clona o descarga este repositorio**

2. **Abre una terminal en la carpeta del proyecto**

3. **Instala las dependencias:**
```bash
pip install -r requirements.txt
```

## ▶️ Cómo ejecutar la aplicación

1. Abre una terminal en la carpeta del proyecto

2. Ejecuta el siguiente comando:
```bash
streamlit run app.py
```

3. La aplicación se abrirá automáticamente en tu navegador (generalmente en `http://localhost:8501`)

## 📚 Módulos de la Aplicación

### 🌐 Riesgo de Mercado

**Juego 1: Ordena los Activos por Volatilidad**
- Analiza gráficos de precios de diferentes activos
- Ordena los activos de menor a mayor riesgo basándote en su comportamiento
- Aprende sobre desviación estándar y volatilidad anualizada

**Juego 2: Shock de Mercado**
- Simula caídas del mercado
- Observa el impacto en tu inversión
- Comprende la importancia de la diversificación

### 💼 Riesgo Financiero (Altman Z-Score)

**Calculadora de Z-Score**
- Ingresa los ratios financieros de una empresa:
  - X1: Capital de trabajo / Total activos
  - X2: Utilidades retenidas / Total activos
  - X3: EBIT / Total activos
  - X4: Patrimonio / Total pasivos
- Calcula el Z'-Score y determina la zona de riesgo

**Juego 1: Encuentra el Ratio Problemático**
- Identifica cuál ratio está afectando negativamente la salud financiera

**Juego 2: Propón la Solución Correcta**
- Elige estrategias adecuadas para mejorar el ratio crítico
- Aprende sobre reestructuración financiera

**Interpretación del Z-Score:**
- Z' > 2.6: Empresa saludable (zona segura 🟢)
- 1.1 ≤ Z' ≤ 2.6: Zona gris (zona de alerta 🟡)
- Z' < 1.1: Alto riesgo de quiebra (zona crítica 🔴)

### 📈 Riesgo Macroeconómico

**Requisitos:**
- Archivo Excel (.xlsx) con columnas:
  - `fecha`: Fecha de la observación
  - `inflacion`: Tasa de inflación (%)
  - `tasa_activa`: Tasa de interés activa de referencia (%)
  - `tasa_pasiva`: Tasa de interés pasiva de referencia (%)

**Dinámica 1: Impacto Macroeconómico en el Z-Score**
- Analiza cómo el entorno económico afecta la salud financiera empresarial
- Predice el comportamiento del Z-Score según condiciones macro

**Dinámica 2: Decisiones de Inversión y Financiamiento**
- Toma decisiones como inversor, empresario y analista bancario
- Evalúa la viabilidad de:
  - Inversiones en depósitos a plazo
  - Endeudamiento empresarial
  - Estrategias de colocación de crédito

## 🎯 Sistema de Puntuación

La aplicación incluye un sistema de gamificación con puntos y niveles:

- **0-30 puntos**: 🌱 Aprendiz de Riesgo
- **31-60 puntos**: 📈 Analista Junior
- **61-90 puntos**: ⭐ Analista Senior
- **91+ puntos**: 🏆 Chief Risk Officer

Los puntos se ganan al:
- Completar juegos correctamente
- Tomar decisiones acertadas
- Identificar correctamente riesgos y soluciones

## 📊 Ejemplo de Archivo Excel para Riesgo Macroeconómico

```excel
fecha         | inflacion | tasa_activa | tasa_pasiva
--------------|-----------|-------------|-------------
2020-01-01    | 2.5       | 9.5         | 5.2
2020-02-01    | 2.7       | 9.8         | 5.3
2020-03-01    | 3.1       | 10.2        | 5.5
...
```

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework para aplicaciones web en Python
- **yfinance**: Descarga de datos financieros
- **pandas**: Manipulación de datos
- **numpy**: Cálculos numéricos
- **plotly**: Visualizaciones interactivas
- **openpyxl**: Lectura de archivos Excel

## 📝 Notas Importantes

1. **Conexión a Internet**: Se requiere para descargar datos de mercado en tiempo real con yfinance

2. **Datos Macroeconómicos**: Debes preparar tu propio archivo Excel con datos históricos de inflación y tasas de interés

3. **Ratios Financieros**: Los ratios del Z-Score deben calcularse previamente desde los estados financieros de la empresa

4. **Persistencia de Datos**: Los puntos y el progreso se mantienen durante la sesión, pero se reinician al cerrar la aplicación

## 🎓 Uso Educativo

Esta aplicación está diseñada para:
- Cursos de finanzas corporativas
- Diplomados en mercado de valores
- Capacitación en gestión de riesgos
- Autoaprendizaje de conceptos financieros

## 📧 Soporte

Para preguntas o sugerencias sobre la aplicación, contacta al departamento académico de la Universidad del Azuay.

## 📄 Licencia

Desarrollado para fines educativos - Universidad del Azuay © 2025

---

¡Disfruta aprendiendo sobre gestión de riesgos! 🎓📊
