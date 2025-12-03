# 📉 Análisis de Value at Risk (VaR) y Crisis Financiera 2008

Aplicación educativa interactiva desarrollada con Streamlit para el análisis de riesgo financiero mediante Value at Risk (VaR) y el estudio de la burbuja inmobiliaria de 2007-2008.

## 🎯 Características

### Análisis de Datos
- **Descarga automática de datos** desde Yahoo Finance
- **Múltiples activos**: SPY, VNQ, BAC, JPM, C, GS, XLF, IYR, y los Magníficos 7 (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META)
- **Frecuencias ajustables**: Diaria, Semanal, Mensual
- **Rendimientos**: Logarítmicos y Simples

### Módulos de Análisis

#### 1. VaR Univariado
- Cálculo paramétrico (distribución normal)
- Cálculo histórico
- Horizontes temporales configurables
- Visualización de distribuciones

#### 2. VaR de Portafolio
- Método de varianzas-covarianzas
- Pesos personalizables o equitativos
- Matriz de correlaciones
- VaR Rolling con ventana móvil

#### 3. Análisis de Crisis
- Comparación por periodos (Pre-crisis, Crisis, Post-crisis)
- Análisis de volatilidad
- Evolución de correlaciones

#### 4. Indicadores Avanzados de Riesgo
- **Curtosis Rolling**: Detección de colas pesadas y eventos extremos
- **VaR Rolling con Alertas**: Sistema de alerta temprana
- **Volatilidad EWMA**: Modelo exponencialmente ponderado (RiskMetrics)

## 🚀 Instalación Local

### Requisitos Previos
- Python 3.8 o superior
- pip

### Pasos de Instalación

1. **Clonar o descargar el repositorio**
```bash
cd var_app
```

2. **Crear un entorno virtual (recomendado)**
```bash
python -m venv .venv
```

3. **Activar el entorno virtual**
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

5. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## ☁️ Despliegue en Streamlit Cloud

### Opción 1: Desde GitHub

1. **Subir el código a GitHub**
   - Crea un repositorio en GitHub
   - Sube los archivos `app.py` y `requirements.txt`

2. **Conectar con Streamlit Cloud**
   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Inicia sesión con tu cuenta de GitHub
   - Haz clic en "New app"
   - Selecciona tu repositorio, rama y archivo `app.py`
   - Haz clic en "Deploy"

### Opción 2: Despliegue Directo

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Sigue las instrucciones para conectar tu repositorio
3. Streamlit detectará automáticamente el `requirements.txt`
4. La app estará disponible en una URL pública

## 📚 Uso Educativo

Esta aplicación está diseñada para estudiantes de finanzas y economía que desean:

- Entender el concepto de Value at Risk
- Analizar el comportamiento de mercados durante crisis
- Explorar indicadores cuantitativos de riesgo
- Aprender sobre diversificación de portafolios
- Estudiar la crisis financiera de 2007-2008

## 📊 Activos Disponibles

### ETFs e Índices
- **SPY**: S&P 500 ETF
- **VNQ**: Real Estate ETF (Sector Inmobiliario)
- **XLF**: Financial Sector ETF
- **IYR**: iShares U.S. Real Estate ETF

### Bancos
- **BAC**: Bank of America
- **JPM**: JP Morgan Chase
- **C**: Citigroup
- **GS**: Goldman Sachs

### Magníficos 7 (Tech)
- **AAPL**: Apple
- **MSFT**: Microsoft
- **GOOGL**: Google (Alphabet)
- **AMZN**: Amazon
- **NVDA**: NVIDIA
- **TSLA**: Tesla
- **META**: Meta (Facebook)

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework de aplicación web
- **pandas**: Manipulación de datos
- **numpy**: Cálculos numéricos
- **yfinance**: Descarga de datos financieros
- **plotly**: Visualizaciones interactivas
- **scipy**: Funciones estadísticas

## 📖 Estructura del Proyecto

```
var_app/
│
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias
├── README.md          # Este archivo
└── .venv/             # Entorno virtual (no subir a git)
```

## 🎓 Créditos

Desarrollado como material educativo para el **Diplomado en Mercado de Valores** de la Universidad del Azuay.

## 📝 Licencia

Este proyecto es de uso educativo. Siéntete libre de usarlo y modificarlo para fines académicos.

## 🤝 Contribuciones

Las sugerencias y mejoras son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## ⚠️ Disclaimer

Esta aplicación es solo para fines educativos. No constituye asesoramiento financiero. Los datos históricos no garantizan resultados futuros. Consulta siempre con un profesional financiero antes de tomar decisiones de inversión.

## 📧 Contacto

Para preguntas o soporte sobre la aplicación, contacta al repositorio del proyecto.

---

**¡Disfruta explorando el mundo del análisis de riesgo financiero! 📈📉**
