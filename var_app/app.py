import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Análisis de VaR y Crisis 2008",
    layout="wide",
    page_icon="📉"
)

# --- FUNCIONES AUXILIARES ---

@st.cache_data
def descargar_datos(tickers, start_date, end_date):
    """Descarga precios de cierre ajustados de Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Si hay múltiples tickers, 'Adj Close' es un nivel del MultiIndex
        if len(tickers) > 1:
            if 'Adj Close' in data.columns.levels[0]:
                data = data['Adj Close']
            else:
                data = data['Close']
        else:
            # Para un solo ticker, la estructura es diferente
            if 'Adj Close' in data.columns:
                data = data[['Adj Close']]
                data.columns = tickers
            elif 'Close' in data.columns:
                data = data[['Close']]
                data.columns = tickers
        
        return data
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return pd.DataFrame()

def calcular_rendimientos(precios, tipo="log"):
    """Calcula rendimientos simples o logarítmicos."""
    if tipo == "log":
        return np.log(precios / precios.shift(1)).dropna()
    else:
        return precios.pct_change().dropna()

def calcular_var_historico(rendimientos, nivel_confianza=0.95):
    """Calcula el VaR histórico."""
    return np.percentile(rendimientos, (1 - nivel_confianza) * 100)

def calcular_var_parametrico(rendimientos, nivel_confianza=0.95):
    """Calcula el VaR paramétrico (distribución normal)."""
    mu = rendimientos.mean()
    sigma = rendimientos.std()
    z = norm.ppf(1 - nivel_confianza)
    return mu + z * sigma

def calcular_var_portafolio(rendimientos, pesos, nivel_confianza=0.95):
    """Calcula el VaR de un portafolio usando varianzas-covarianzas."""
    cov_matrix = rendimientos.cov()
    var_portafolio = np.dot(pesos.T, np.dot(cov_matrix, pesos))
    sigma_portafolio = np.sqrt(var_portafolio)
    mu_portafolio = np.sum(rendimientos.mean() * pesos)
    z = norm.ppf(1 - nivel_confianza)
    var_valor = mu_portafolio + z * sigma_portafolio
    return var_valor, sigma_portafolio, cov_matrix

def calcular_kurtosis_rolling(rendimientos, ventana=60):
    """Calcula la curtosis rolling de los rendimientos."""
    from scipy.stats import kurtosis
    kurtosis_rolling = rendimientos.rolling(window=ventana).apply(lambda x: kurtosis(x, fisher=False), raw=True)
    return kurtosis_rolling

def calcular_var_rolling(rendimientos, ventana=250, nivel_confianza=0.95):
    """Calcula el VaR rolling usando percentiles históricos."""
    var_rolling = rendimientos.rolling(window=ventana).quantile(1 - nivel_confianza)
    return var_rolling

def calcular_stress_ratio(rendimientos, ventana_sigma=30, ventana_var=250):
    """Calcula el Stress Ratio: sigma_rolling / |VaR_rolling|."""
    sigma_rolling = rendimientos.rolling(window=ventana_sigma).std()
    var_rolling = calcular_var_rolling(rendimientos, ventana=ventana_var)
    stress_ratio = sigma_rolling / np.abs(var_rolling)
    return stress_ratio, sigma_rolling, var_rolling

def calcular_ewma_volatilidad(rendimientos, lambda_param=0.94):
    """Calcula la volatilidad EWMA (Exponentially Weighted Moving Average)."""
    # Inicializar con la varianza histórica
    var_inicial = rendimientos.var()
    ewma_var = [var_inicial]
    
    for ret in rendimientos[1:]:
        nueva_var = lambda_param * ewma_var[-1] + (1 - lambda_param) * (ret ** 2)
        ewma_var.append(nueva_var)
    
    ewma_vol = pd.Series(np.sqrt(ewma_var), index=rendimientos.index)
    return ewma_vol

def interpretar_kurtosis(kurt_value):
    """Devuelve interpretación y color según el nivel de kurtosis."""
    if kurt_value <= 3:
        return "Normal", "green"
    elif kurt_value <= 5:
        return "Colas pesadas comienzan", "yellow"
    elif kurt_value <= 10:
        return "⚠️ ALERTA", "orange"
    else:
        return "🚨 RIESGO SISTÉMICO INMINENTE", "red"

def interpretar_stress_ratio(sr_value):
    """Devuelve interpretación y color según el Stress Ratio."""
    if sr_value < 0.3:
        return "🟢 Normal", "green"
    elif sr_value <= 0.6:
        return "🟡 Tensión moderada", "orange"
    else:
        return "🔴 Riesgo sistémico", "red"

# --- INTERFAZ DE USUARIO ---

# Título Principal
st.title("📉 Análisis de Value at Risk (VaR) y la Burbuja Inmobiliaria 2007–2008")
st.markdown("""
Esta aplicación educativa permite explorar el concepto de **Value at Risk (VaR)** y analizar cómo se comportaron 
los mercados financieros durante la crisis subprime.
""")

# --- BARRA LATERAL ---
# Logo en el sidebar
try:
    st.sidebar.image("Logo BVQ Color.png", width=250)
    st.sidebar.markdown("---")
except:
    pass

st.sidebar.header("⚙️ Configuración")

# Selección de Fechas
start_date = st.sidebar.date_input("Fecha de inicio", pd.to_datetime("2004-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", pd.to_datetime("2025-12-31"))

# Selección de Activos
tickers_default = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
tickers = st.sidebar.multiselect("Selecciona los activos", 
                                 ["SPY", "VNQ", "BAC", "JPM", "C", "GS", "XLF", "IYR",
                                  "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"], 
                                 default=tickers_default)

# Frecuencia
frecuencia = st.sidebar.selectbox("Frecuencia de datos", ["Diaria", "Semanal", "Mensual"])
intervalo_map = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}

# Descarga de datos
if not tickers:
    st.warning("Por favor selecciona al menos un activo.")
    st.stop()

with st.spinner('Descargando datos de mercado...'):
    # yfinance no soporta re-sampling directo en la descarga para weekly/monthly de forma robusta con fechas exactas a veces,
    # pero podemos descargar diario y resamplear nosotros o usar el intervalo de yf.
    # Para simplicidad y consistencia, descargamos diario y resampleamos si es necesario.
    datos_raw = descargar_datos(tickers, start_date, end_date)
    
    if datos_raw.empty:
        st.error("No se encontraron datos para los activos seleccionados en el rango de fechas.")
        st.stop()

    if frecuencia == "Semanal":
        datos = datos_raw.resample('W').last()
    elif frecuencia == "Mensual":
        datos = datos_raw.resample('M').last()
    else:
        datos = datos_raw

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Introducción", 
    "Datos y precios", 
    "VaR univariado", 
    "VaR de portafolio", 
    "Burbuja inmobiliaria", 
    "Indicadores Avanzados de Riesgo",
    "Actividad"
])

# --- 1. INTRODUCCIÓN ---
with tab1:
    st.header("📘 Introducción al Value at Risk (VaR)")
    
    st.markdown("""
    ### ¿Qué es el VaR?
    El **Value at Risk (VaR)** es una medida estadística utilizada para cuantificar el nivel de riesgo financiero 
    dentro de una empresa o cartera de inversiones en un marco de tiempo específico.
    
    Básicamente, responde a la pregunta: 
    > *"¿Cuánto es lo máximo que puedo esperar perder con un nivel de confianza dado (por ejemplo, 95%) en un periodo determinado?"*
    
    ### VaR Univariado vs. VaR de Portafolio
    *   **VaR Univariado:** Se calcula para un solo activo. Analiza la distribución de rendimientos de ese activo individual.
    *   **VaR de Portafolio:** Considera múltiples activos. Aquí es crucial la **correlación** entre activos. 
        Si los activos no están perfectamente correlacionados, el riesgo del portafolio (diversificado) suele ser menor 
        que la suma de los riesgos individuales.
    
    ### La Burbuja Inmobiliaria 2007–2008
    La crisis financiera de 2007-2008 fue desatada por el colapso de la burbuja inmobiliaria en Estados Unidos. 
    Los bancos habían otorgado hipotecas de alto riesgo (subprime) que luego empaquetaron en productos financieros complejos.
    
    Cuando los precios de las viviendas cayeron y los impagos aumentaron, estos activos se volvieron tóxicos, 
    llevando a la quiebra a grandes instituciones como Lehman Brothers y afectando severamente a bancos como 
    Bank of America (BAC) y JP Morgan (JPM), así como al sector inmobiliario (VNQ).
    """)

# --- 2. DATOS Y PRECIOS ---
with tab2:
    st.header("📊 Datos Históricos y Precios")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        tipo_retorno = st.radio("Tipo de rendimientos", ["Logarítmicos", "Simples"])
        tipo_ret_code = "log" if tipo_retorno == "Logarítmicos" else "simple"
    
    # Calcular rendimientos
    rendimientos = calcular_rendimientos(datos, tipo=tipo_ret_code)
    
    st.subheader("Evolución de Precios")
    fig_precios = px.line(datos, x=datos.index, y=datos.columns, title="Precios Ajustados de Cierre")
    st.plotly_chart(fig_precios, use_container_width=True)
    
    st.subheader("Evolución de Rendimientos")
    fig_retornos = px.line(rendimientos, x=rendimientos.index, y=rendimientos.columns, title=f"Rendimientos {tipo_retorno}")
    st.plotly_chart(fig_retornos, use_container_width=True)
    
    with st.expander("Ver tabla de precios"):
        st.dataframe(datos)

# --- 3. VAR UNIVARIADO ---
with tab3:
    st.header("📉 VaR Univariado")
    
    col_var1, col_var2 = st.columns([1, 3])
    
    with col_var1:
        activo_uni = st.selectbox("Selecciona un activo", tickers)
        confianza_uni = st.selectbox("Nivel de confianza", [0.95, 0.99])
        metodo_var = st.radio("Método de cálculo", ["Paramétrico (Normal)", "Histórico"])
        horizonte = st.number_input("Horizonte temporal (días)", min_value=1, value=1)
    
    rets_activo = rendimientos[activo_uni]
    
    if metodo_var == "Paramétrico (Normal)":
        var_val = calcular_var_parametrico(rets_activo, confianza_uni)
    else:
        var_val = calcular_var_historico(rets_activo, confianza_uni)
    
    # Escalar VaR por horizonte (asumiendo raíz del tiempo para paramétrico, simple para propósitos didácticos)
    # Nota: Para histórico puro, escalar es más complejo, aquí usamos la regla de raíz de t para simplificar la didáctica
    var_val_horizonte = var_val * np.sqrt(horizonte)
    
    with col_var2:
        st.metric(label=f"VaR {confianza_uni*100:.0f}% ({horizonte} días)", value=f"{var_val_horizonte:.2%}")
        
        st.info(f"""
        **Interpretación:** 
        Con un nivel de confianza del {confianza_uni*100:.0f}%, se estima que la pérdida máxima del activo **{activo_uni}** 
        en un periodo de {horizonte} día(s) no superará el **{abs(var_val_horizonte):.2%}**.
        
        En términos monetarios, si inviertes $10,000, hay un {confianza_uni*100:.0f}% de probabilidad de que tu pérdida 
        no exceda **${10000 * abs(var_val_horizonte):.2f}**.
        """)
        
        # Histograma
        fig_hist = px.histogram(rets_activo, nbins=50, title=f"Distribución de Rendimientos: {activo_uni}", 
                                labels={'value': 'Rendimiento'}, opacity=0.7)
        fig_hist.add_vline(x=var_val, line_dash="dash", line_color="red", annotation_text=f"VaR (1 día): {var_val:.2%}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Estadísticas
        stats_df = pd.DataFrame({
            "Métrica": ["Media", "Desviación Estándar", "Mínimo", "Máximo"],
            "Valor": [rets_activo.mean(), rets_activo.std(), rets_activo.min(), rets_activo.max()]
        })
        st.table(stats_df.style.format({"Valor": "{:.4%}"}))

# --- 4. VAR DE PORTAFOLIO ---
with tab4:
    st.header("💼 VaR de Portafolio (Varianzas-Covarianzas)")
    
    activos_port = st.multiselect("Activos del portafolio", tickers, default=tickers)
    
    if not activos_port:
        st.warning("Selecciona activos para el portafolio.")
    else:
        col_p1, col_p2 = st.columns([1, 2])
        
        with col_p1:
            tipo_pesos = st.radio("Asignación de pesos", ["Equitativo", "Manual"])
            pesos = []
            if tipo_pesos == "Equitativo":
                pesos = np.array([1/len(activos_port)] * len(activos_port))
                st.write("Pesos asignados automáticamente:")
                for a, p in zip(activos_port, pesos):
                    st.write(f"- {a}: {p:.2%}")
            else:
                st.write("Ingresa los pesos (deben sumar 1):")
                pesos_input = []
                for a in activos_port:
                    p = st.number_input(f"Peso para {a}", min_value=0.0, max_value=1.0, value=1.0/len(activos_port), step=0.05)
                    pesos_input.append(p)
                pesos = np.array(pesos_input)
                total_pesos = sum(pesos)
                if not np.isclose(total_pesos, 1.0):
                    st.error(f"⚠️ Los pesos suman {total_pesos:.2f}, deben sumar 1.0")
                else:
                    st.success("Pesos correctos.")

        if np.isclose(sum(pesos), 1.0) or tipo_pesos == "Equitativo":
            rets_port = rendimientos[activos_port]
            
            # Cálculo
            var_port_95, sigma_port, cov_mat = calcular_var_portafolio(rets_port, pesos, 0.95)
            var_port_99, _, _ = calcular_var_portafolio(rets_port, pesos, 0.99)
            
            with col_p2:
                st.subheader("Resultados del Portafolio")
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("VaR 95% (1 día)", f"{var_port_95:.2%}")
                col_res2.metric("VaR 99% (1 día)", f"{var_port_99:.2%}")
                st.metric("Volatilidad Anualizada (aprox)", f"{sigma_port * np.sqrt(252):.2%}")
                
                # Matriz de Correlación
                st.subheader("Matriz de Correlaciones")
                corr_matrix = rets_port.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)

            # Rolling VaR
            st.subheader("Evolución del VaR del Portafolio (Rolling Window)")
            window = st.slider("Ventana móvil (días)", 30, 500, 252)
            
            # Calculamos el retorno del portafolio histórico
            portfolio_returns = (rets_port * pesos).sum(axis=1)
            
            # Rolling VaR
            rolling_mean = portfolio_returns.rolling(window=window).mean()
            rolling_std = portfolio_returns.rolling(window=window).std()
            rolling_var_95 = rolling_mean + norm.ppf(0.05) * rolling_std
            
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns*100, name="Retorno Diario", opacity=0.3))
            fig_rolling.add_trace(go.Scatter(x=rolling_var_95.index, y=rolling_var_95*100, name=f"VaR 95% ({window} días)", line=dict(color='red')))
            fig_rolling.update_layout(
                title="Retornos del Portafolio vs VaR Histórico Móvil",
                yaxis_title="Porcentaje (%)",
                yaxis=dict(tickformat=".2f", ticksuffix="%")
            )
            st.plotly_chart(fig_rolling, use_container_width=True)

# --- 5. BURBUJA INMOBILIARIA ---
with tab5:
    st.header("🏚️ Análisis de la Crisis Subprime (2007-2008)")
    
    periodo = st.radio("Selecciona el periodo de análisis:", 
                       ["Pre-crisis (2004-2006)", "Crisis (2007-2009)", "Post-crisis (2010-2012)"],
                       horizontal=True)
    
    if periodo == "Pre-crisis (2004-2006)":
        p_start, p_end = "2004-01-01", "2006-12-31"
    elif periodo == "Crisis (2007-2009)":
        p_start, p_end = "2007-01-01", "2009-12-31"
    else:
        p_start, p_end = "2010-01-01", "2012-12-31"
        
    st.markdown(f"**Analizando periodo:** {p_start} al {p_end}")
    
    # Filtrar datos por periodo
    mask = (rendimientos.index >= p_start) & (rendimientos.index <= p_end)
    rets_periodo = rendimientos.loc[mask]
    
    if rets_periodo.empty:
        st.error("No hay datos suficientes en este periodo para los activos seleccionados.")
    else:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("Volatilidad por Activo")
            vols = rets_periodo.std()
            fig_vol = px.bar(vols, title="Desviación Estándar (Riesgo)", labels={'value': 'Volatilidad', 'index': 'Activo'})
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with col_c2:
            st.subheader("Correlaciones en el Periodo")
            corr_periodo = rets_periodo.corr()
            fig_corr_p = px.imshow(corr_periodo, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr_p, use_container_width=True)
            
        st.subheader("Observaciones Clave")
        st.info("""
        **Lo que debes observar:**
        1.  **Aumento de Volatilidad:** Durante la crisis, las barras de volatilidad (especialmente en bancos y sector inmobiliario) deberían ser mucho más altas.
        2.  **Aumento de Correlaciones:** En tiempos de pánico, "todo cae junto". Las correlaciones tienden a acercarse a 1, reduciendo el beneficio de la diversificación.
        3.  **VaR más profundo:** El riesgo de pérdida extrema aumenta significativamente.
        """)

# --- 6. INDICADORES AVANZADOS DE RIESGO ---
with tab6:
    st.header("🎯 Indicadores Avanzados de Riesgo")
    
    st.markdown("""
    Esta sección presenta indicadores cuantitativos avanzados para detectar **vulnerabilidad financiera** 
    antes de que ocurra una crisis sistémica. Estos indicadores son especialmente útiles para:
    - Anticipar períodos de alta volatilidad
    - Identificar acumulación de riesgo sistémico
    - Detectar anomalías en la distribución de rendimientos
    """)
    
    # Configuración para indicadores avanzados
    activos_ind = st.multiselect("Activos para análisis avanzado", tickers, default=tickers, key="ind_avanzados")
    
    if not activos_ind:
        st.warning("Selecciona al menos un activo para el análisis.")
    else:
        # Configurar pesos del portafolio
        pesos_ind = np.array([1/len(activos_ind)] * len(activos_ind))
        rets_ind = rendimientos[activos_ind]
        portfolio_returns_ind = (rets_ind * pesos_ind).sum(axis=1)
        
        # --- 1. ROLLING KURTOSIS ---
        st.subheader("📊 1. Curtosis Rolling (Rolling Kurtosis)")
        st.markdown("""
        La **curtosis** mide el "grosor" de las colas de la distribución de rendimientos. 
        Valores altos indican mayor probabilidad de eventos extremos (crashes o rallies).
        """)
        
        ventana_kurt = st.slider("Ventana para curtosis (días)", 30, 120, 60, key="ventana_kurt")
        kurtosis_roll = calcular_kurtosis_rolling(portfolio_returns_ind, ventana=ventana_kurt)
        
        # Gráfico de Kurtosis
        fig_kurt = go.Figure()
        fig_kurt.add_trace(go.Scatter(
            x=kurtosis_roll.index, 
            y=kurtosis_roll, 
            name="Curtosis Rolling",
            line=dict(color='purple', width=2)
        ))
        
        # Líneas de referencia
        fig_kurt.add_hline(y=3, line_dash="dash", line_color="green", annotation_text="Normal (3)")
        fig_kurt.add_hline(y=5, line_dash="dash", line_color="yellow", annotation_text="Colas pesadas (5)")
        fig_kurt.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Alerta (10)")
        fig_kurt.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Riesgo sistémico (15)")
        
        fig_kurt.update_layout(
            title=f"Curtosis Rolling del Portafolio (ventana {ventana_kurt} días)",
            xaxis_title="Fecha",
            yaxis_title="Curtosis",
            hovermode='x unified'
        )
        st.plotly_chart(fig_kurt, use_container_width=True)
        
        # Interpretación actual
        kurt_actual = kurtosis_roll.dropna().iloc[-1] if not kurtosis_roll.dropna().empty else 3
        interpretacion, color = interpretar_kurtosis(kurt_actual)
        
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Curtosis Actual", f"{kurt_actual:.2f}")
        col_k2.markdown(f"**Estado:** <span style='color:{color}; font-size:20px'>{interpretacion}</span>", unsafe_allow_html=True)
        
        with st.expander("📖 Guía de interpretación"):
            st.markdown("""
            - **≤ 3**: Distribución normal, riesgo moderado
            - **4-5**: Comienzan a aparecer colas pesadas, mayor riesgo de eventos extremos
            - **8-10**: ⚠️ ALERTA - Alta probabilidad de movimientos extremos
            - **> 15**: 🚨 RIESGO SISTÉMICO INMINENTE - Probabilidad muy alta de crash
            
            Durante la crisis de 2008, la curtosis se disparó por encima de 15 en múltiples ocasiones.
            """)
        
        # --- 2. VAR ROLLING ALERT ---
        st.subheader("⚡ 2. VaR Rolling con Alerta Temprana")
        st.markdown("""
        El **VaR Rolling** muestra cómo evoluciona el riesgo a lo largo del tiempo. 
        Cuando el VaR actual supera significativamente su promedio histórico, es señal de alerta.
        """)
        
        ventana_var_roll = st.slider("Ventana para VaR (días)", 100, 500, 250, key="ventana_var")
        var_rolling = calcular_var_rolling(portfolio_returns_ind, ventana=ventana_var_roll, nivel_confianza=0.95)
        
        # Calcular promedio de 2 años previos (504 días hábiles aprox)
        var_promedio_2y = var_rolling.rolling(window=504).mean()
        alerta_var = np.abs(var_rolling) > 2 * np.abs(var_promedio_2y)
        
        # Gráfico VaR Rolling
        fig_var = go.Figure()
        fig_var.add_trace(go.Scatter(
            x=var_rolling.index, 
            y=var_rolling * 100, 
            name="VaR Rolling 95%",
            line=dict(color='darkred', width=2)
        ))
        fig_var.add_trace(go.Scatter(
            x=var_promedio_2y.index, 
            y=var_promedio_2y * 100, 
            name="Promedio 2 años",
            line=dict(color='blue', width=1, dash='dot')
        ))
        
        # Marcar zonas de alerta
        alertas_fechas = var_rolling.index[alerta_var]
        if len(alertas_fechas) > 0:
            fig_var.add_trace(go.Scatter(
                x=alertas_fechas,
                y=(var_rolling[alerta_var] * 100),
                mode='markers',
                name='🚨 Alerta',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        fig_var.update_layout(
            title=f"VaR Rolling 95% (ventana {ventana_var_roll} días)",
            xaxis_title="Fecha",
            yaxis_title="VaR (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_var, use_container_width=True)
        
        # Verificar alerta actual
        if not var_rolling.dropna().empty and not var_promedio_2y.dropna().empty:
            var_actual = var_rolling.dropna().iloc[-1]
            var_prom_actual = var_promedio_2y.dropna().iloc[-1]
            
            if np.abs(var_actual) > 2 * np.abs(var_prom_actual):
                st.error(f"""
                🚨 **ALERTA TEMPRANA ACTIVADA**
                
                El VaR actual ({var_actual:.2%}) supera el doble del promedio de los últimos 2 años ({var_prom_actual:.2%}).
                Esto indica un aumento significativo del riesgo de mercado.
                """)
            else:
                st.success("✅ VaR dentro de niveles normales")
        
        # --- 3. EWMA VOLATILITY ---
        st.subheader("📈 3. Volatilidad EWMA (Exponentially Weighted)")
        st.markdown("""
        La **volatilidad EWMA** da más peso a los datos recientes, permitiendo detectar 
        cambios en la volatilidad más rápidamente que las medias móviles simples.
        
        Fórmula: `σ²ₜ = λ × σ²ₜ₋₁ + (1-λ) × r²ₜ`
        """)
        
        lambda_param = st.slider("Parámetro λ (decay factor)", 0.85, 0.98, 0.94, 0.01, key="lambda")
        ewma_vol = calcular_ewma_volatilidad(portfolio_returns_ind, lambda_param=lambda_param)
        
        # Calcular promedio histórico y alerta (50% por encima)
        ewma_promedio = ewma_vol.mean()
        umbral_alerta = ewma_promedio * 1.5
        alerta_ewma = ewma_vol > umbral_alerta
        
        # Gráfico EWMA
        fig_ewma = go.Figure()
        fig_ewma.add_trace(go.Scatter(
            x=ewma_vol.index,
            y=ewma_vol * 100,
            name="Volatilidad EWMA",
            line=dict(color='teal', width=2)
        ))
        fig_ewma.add_hline(
            y=ewma_promedio * 100, 
            line_dash="dash", 
            line_color="blue", 
            annotation_text=f"Promedio histórico ({ewma_promedio*100:.2f}%)"
        )
        fig_ewma.add_hline(
            y=umbral_alerta * 100, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Umbral alerta (+50%): {umbral_alerta*100:.2f}%"
        )
        
        # Marcar zonas de alerta
        alertas_ewma_fechas = ewma_vol.index[alerta_ewma]
        if len(alertas_ewma_fechas) > 0:
            fig_ewma.add_trace(go.Scatter(
                x=alertas_ewma_fechas,
                y=(ewma_vol[alerta_ewma] * 100),
                mode='markers',
                name='⚠️ Alerta alta volatilidad',
                marker=dict(color='red', size=6)
            ))
        
        fig_ewma.update_layout(
            title=f"Volatilidad EWMA (λ={lambda_param})",
            xaxis_title="Fecha",
            yaxis_title="Volatilidad (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_ewma, use_container_width=True)
        
        # Métricas y alerta
        if not ewma_vol.dropna().empty:
            ewma_actual = ewma_vol.dropna().iloc[-1]
            
            col_ew1, col_ew2, col_ew3 = st.columns(3)
            col_ew1.metric("Volatilidad EWMA Actual", f"{ewma_actual*100:.2f}%")
            col_ew2.metric("Promedio Histórico", f"{ewma_promedio*100:.2f}%")
            col_ew3.metric("Umbral Alerta", f"{umbral_alerta*100:.2f}%")
            
            if ewma_actual > umbral_alerta:
                st.warning(f"""
                ⚠️ **ALERTA DE VOLATILIDAD**
                
                La volatilidad EWMA actual ({ewma_actual*100:.2f}%) supera en más del 50% 
                su promedio histórico. Esto indica un régimen de alta volatilidad.
                """)
            else:
                st.info("✅ Volatilidad dentro de rangos normales")
        
        with st.expander("📖 Acerca de EWMA"):
            st.markdown(f"""
            **Parámetro λ = {lambda_param}**
            - λ alto (0.94-0.97): Más peso a historia reciente, responde lento
            - λ bajo (0.85-0.90): Reacciona rápido a cambios de volatilidad
            
            RiskMetrics™ de JP Morgan recomienda λ=0.94 para datos diarios.
            
            **Ventajas de EWMA:**
            - Detecta cambios de régimen de volatilidad más rápido
            - No requiere ventana fija como rolling std
            - Usado en modelos de riesgo profesionales (GARCH, etc.)
            """)

# --- 7. ACTIVIDAD GRUPAL ---
with tab7:
    st.header("📝 Actividad Grupal")
    
    # Solicitar código de acceso
    codigo = st.text_input("Ingresa el código para ver la actividad:", type="password")
    
    if codigo == "datos2025":
        # Mostrar contenido completo de la actividad
        st.markdown("""
# Actividad Grupal: Detectores de Crisis con VaR, Curtosis y EWMA

Trabajo en grupos de 3 a 4 personas utilizando esta misma aplicación de Streamlit.

## Configuración inicial en la App

Configuren la app de la siguiente manera:

1. Activos seleccionados:
   - SPY
   - VNQ
   - BAC
   - JPM

2. Frecuencia: Diaria  
3. Tipo de rendimientos: Logarítmicos  
4. Rango de fechas: desde 2004 hasta 2025  

5. Parámetros iniciales:
   - Ventana de curtosis: 60 días
   - Ventana del VaR rolling: 250 días
   - Parámetro lambda (EWMA): 0.94
   - Horizonte del VaR univariado: 1 día

Asegúrense de que todos en el grupo trabajan con la misma configuración inicial.

## Parte 1: Exploración con Sliders

El objetivo de esta parte es entender cómo los parámetros que se controlan con los sliders
afectan las señales de riesgo.

A) Ventana de curtosis (por ejemplo, entre 30 y 120 días)

- Cambien la ventana de curtosis a valores más pequeños y más grandes.
- Observen cómo cambian los picos de curtosis en periodos como 2008, 2011, 2020 y 2025.
- Respondan:
  - ¿Con una ventana pequeña, la curtosis reacciona más rápido o más lento a las crisis?
  - ¿Con una ventana grande, se suavizan demasiado las señales?

B) Ventana del VaR rolling (por ejemplo, entre 50 y 360 días)

- Modifiquen la ventana del VaR rolling.
- Observen cómo cambia la forma de la serie de VaR a lo largo del tiempo.
- Respondan:
  - ¿Con ventanas cortas aparecen más "falsas alarmas"?
  - ¿Con ventanas largas se pierde detalle de ciertos episodios de riesgo?

C) Parámetro lambda de EWMA (por ejemplo, entre 0.90 y 0.99)

- Cambien el valor de lambda y observen la volatilidad EWMA.
- Respondan:
  - ¿Qué ocurre cuando lambda es más bajo (por ejemplo 0.90)?
  - ¿Qué ocurre cuando lambda es más alto (por ejemplo 0.99)?
  - ¿Con qué lambda sienten que mejor se identifican los periodos de tensión fuerte
    sin generar demasiadas falsas alarmas?

D) Horizonte del VaR univariado (por ejemplo, entre 1 y 10 días)

- Cambien el horizonte del VaR univariado.
- Comparen el VaR para 1 día, 5 días y 10 días.
- Respondan:
  - ¿Cómo cambia el tamaño del VaR cuando aumenta el horizonte?
  - ¿Les parece razonable que el riesgo crezca aproximadamente con la raíz del tiempo?

## Parte 2: Detección de Crisis

Usando las gráficas de VaR, curtosis, volatilidad EWMA y correlaciones, respondan:

1. Crisis financiera de 2008
   - ¿En qué periodo ven un aumento fuerte en la curtosis?
   - ¿Cuándo el VaR rolling se vuelve claramente más negativo?
   - ¿En qué momento la volatilidad EWMA muestra un salto significativo?
   - ¿Las correlaciones entre SPY, VNQ, BAC y JPM se acercan a valores altos
     (por ejemplo, mayores a 0.8)?

2. Crisis de deuda europea (alrededor de 2011)
   - ¿Qué indicadores muestran señales claras en ese periodo?
   - ¿Algún parámetro de los sliders ayuda a ver mejor esas señales?

3. Episodio COVID-19 (alrededor de 2020)
   - ¿Cómo reaccionan la curtosis, el VaR rolling y la volatilidad EWMA?
   - ¿Cuál de estos indicadores parece reaccionar primero?

4. Episodios recientes (alrededor de 2024 y 2025)
   - ¿La app muestra señales de estrés financiero en esos años?
   - ¿Esas señales se deben a caídas, subidas fuertes o movimientos extremos en ambos sentidos?

## Parte 3: Informe del Grupo

En una hoja o documento, el grupo debe resumir:

1. Qué parámetro (ventana de curtosis, ventana de VaR rolling, lambda de EWMA u horizonte
   del VaR) les pareció más útil para identificar cambios de régimen de riesgo.

2. Qué combinación de parámetros recomendarían como configuración estándar en esta app
   para analizar crisis financieras. Por ejemplo:
   - Ventana de curtosis recomendada
   - Ventana del VaR rolling recomendada
   - Valor de lambda recomendado
   - Horizonte de VaR recomendado

3. Qué indicador consideran más adelantado para detectar tensión en el mercado:
   - Curtosis (colas pesadas)
   - Volatilidad EWMA (aceleración de la volatilidad)
   - VaR rolling (fragilidad del portafolio)
   - Correlaciones entre activos (pérdida de diversificación)

4. Si hubieran estado en un comité de riesgos antes de la crisis de 2008, ¿con la evidencia
   de estos indicadores habrían recomendado tomar alguna acción preventiva? Expliquen cuál.

## Parte 4: Puesta en Común

Cada grupo debe estar preparado para compartir brevemente:

- La combinación de parámetros que eligieron como recomendada.
- El indicador que consideraron más relevante.
- Una idea clave que hayan aprendido sobre cómo se manifiesta una crisis financiera
  en estos indicadores cuantitativos.
        """)
        
        # Botón de descarga de datos
        st.markdown("---")
        st.subheader("Descarga de Datos")
        st.markdown("Si deseas realizar tu propio análisis en Excel o Python, puedes descargar los datos procesados aquí:")
        
        csv = rendimientos.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Descargar Rendimientos (CSV)",
            data=csv,
            file_name='rendimientos_var_app.csv',
            mime='text/csv',
        )
    elif codigo == "":
        st.info("Ingresa el código proporcionado por el profesor para ver la actividad.")
    else:
        st.error("Código incorrecto. Ingresa el código proporcionado por el profesor para ver la actividad.")

