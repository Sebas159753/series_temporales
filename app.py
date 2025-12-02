"""
Laboratorio - Diplomado de mercado de valores y estrategias de inversión
Aplicación didáctica para aprender sobre riesgos financieros
Autor: Bolsa de Valores Quito
Fecha: 2025
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

st.set_page_config(
    page_title="Laboratorio - Diplomado de mercado de valores y estrategias de inversión",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session_state para puntos
if 'puntos_mercado' not in st.session_state:
    st.session_state.puntos_mercado = 0
if 'puntos_financiero' not in st.session_state:
    st.session_state.puntos_financiero = 0
if 'puntos_macro' not in st.session_state:
    st.session_state.puntos_macro = 0

# Función para calcular puntos totales
def calcular_puntos_totales():
    return st.session_state.puntos_mercado + st.session_state.puntos_financiero + st.session_state.puntos_macro

# Función para determinar nivel
def obtener_nivel(puntos):
    if puntos >= 91:
        return "🏆 Chief Risk Officer", "#FFD700"
    elif puntos >= 61:
        return "⭐ Analista Senior", "#C0C0C0"
    elif puntos >= 31:
        return "📈 Analista Junior", "#CD7F32"
    else:
        return "🌱 Aprendiz de Riesgo", "#90EE90"

# ============================================================================
# HEADER Y SISTEMA DE PUNTOS GLOBAL
# ============================================================================

st.title("🎓 Laboratorio - Diplomado de mercado de valores y estrategias de inversión")
st.markdown("### Aprende sobre gestión de riesgos de forma interactiva")

# Mostrar puntos globales en el sidebar
with st.sidebar:
    # Logo de BVQ
    st.image("Logo BVQ Color.png", use_container_width=True)
    st.divider()
    
    st.header("📊 Tu Progreso")
    puntos_totales = calcular_puntos_totales()
    nivel, color = obtener_nivel(puntos_totales)
    
    st.metric("Puntos Totales", puntos_totales)
    st.markdown(f"<h3 style='color: {color};'>{nivel}</h3>", unsafe_allow_html=True)
    
    st.divider()
    st.subheader("Puntos por Módulo:")
    st.write(f"🌐 Riesgo de Mercado: {st.session_state.puntos_mercado}")
    st.write(f"💼 Riesgo Financiero: {st.session_state.puntos_financiero}")
    st.write(f"📈 Riesgo Macroeconómico: {st.session_state.puntos_macro}")
    
    st.divider()
    st.info("💡 **Tip:** Completa todos los juegos para maximizar tus puntos y alcanzar el nivel de Chief Risk Officer")

# ============================================================================
# PESTAÑA 1: RIESGO DE MERCADO
# ============================================================================

def tab_riesgo_mercado():
    st.header("🌐 Riesgo de Mercado")
    st.markdown("""
    El **riesgo de mercado** se refiere a la posibilidad de pérdidas en el valor de los activos 
    debido a cambios en los precios del mercado. La volatilidad (desviación estándar de los retornos) 
    es una medida clave del riesgo de mercado.
    """)
    
    # Lista de tickers disponibles
    tickers = ['SPY', 'QQQ', 'TSLA', 'BTC-USD', 'GLD', 'AAPL']
    
    # Descargar volatilidades de TODOS los tickers (solo una vez)
    if 'volatilidades_all' not in st.session_state:
        with st.spinner("Calculando volatilidades de todos los activos..."):
            st.session_state.volatilidades_all = {}
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(period="1y")
                    
                    if not data.empty and len(data) > 10:
                        precios = data['Close'].copy().dropna()
                        
                        if len(precios) > 10:
                            retornos = precios.pct_change().dropna()
                            if len(retornos) > 0:
                                vol_value = retornos.std() * np.sqrt(252) * 100
                                st.session_state.volatilidades_all[ticker] = float(vol_value.iloc[0]) if hasattr(vol_value, 'iloc') else float(vol_value)
                except Exception as e:
                    pass  # Silenciar errores en la carga inicial
    
    # Usar las volatilidades globales
    volatilidades_globales = st.session_state.volatilidades_all
    
    # Selección de periodo
    col1, col2 = st.columns(2)
    with col1:
        fecha_fin = datetime.now()
        fecha_inicio = fecha_fin - timedelta(days=365)
    
    # ========================================================================
    # JUEGO 1: ORDENA LOS ACTIVOS POR VOLATILIDAD
    # ========================================================================
    
    st.subheader("🎯 Juego 1: Ordena los Activos por Volatilidad")
    st.markdown("Observa los gráficos de precios y ordena los activos de **menor** a **mayor** riesgo (volatilidad).")
    
    # Descargar datos
    with st.spinner("Descargando datos del mercado..."):
        datos = {}
        volatilidades = {}
        
        # Seleccionar 4 activos aleatorios para este juego
        import random
        if 'tickers_juego1' not in st.session_state:
            st.session_state.tickers_juego1 = random.sample(tickers, 4)
        
        tickers_juego = st.session_state.tickers_juego1
        
        for ticker in tickers_juego:
            try:
                # Descargar datos usando Ticker object (más confiable)
                stock = yf.Ticker(ticker)
                data = stock.history(period="1y")  # Último año
                
                if not data.empty and len(data) > 10:
                    # Obtener columna de precios de cierre
                    precios = data['Close'].copy()
                    
                    # Limpiar datos
                    precios = precios.dropna()
                    
                    if len(precios) > 10:
                        datos[ticker] = precios
                        retornos = precios.pct_change().dropna()
                        if len(retornos) > 0:
                            # Calcular volatilidad anualizada
                            vol_value = retornos.std() * np.sqrt(252) * 100
                            volatilidades[ticker] = float(vol_value.iloc[0]) if hasattr(vol_value, 'iloc') else float(vol_value)
                        st.success(f"✅ {ticker}: {len(precios)} días de datos descargados")
                    else:
                        st.warning(f"⚠️ {ticker}: Datos insuficientes ({len(precios)} días)")
                else:
                    st.warning(f"⚠️ {ticker}: No se obtuvieron datos")
            except Exception as e:
                st.error(f"❌ Error con {ticker}: {str(e)}")
        
        if len(datos) == 0:
            st.error("❌ No se pudo descargar ningún dato. Verifica tu conexión a internet.")
            st.info("💡 **Sugerencia:** Intenta recargar la página o verifica que tienes acceso a Yahoo Finance.")
    
    if len(datos) >= 3:
        # Crear mapeo de letras (A, B, C, D) a tickers
        letras = ['A', 'B', 'C', 'D']
        if 'mapeo_activos' not in st.session_state:
            st.session_state.mapeo_activos = dict(zip(letras[:len(datos)], list(datos.keys())))
        
        mapeo = st.session_state.mapeo_activos
        
        # Mostrar gráficos sin revelar el ticker - en formato 2x2
        st.markdown("**Observa los gráficos de precios:**")
        
        letras_list = list(mapeo.keys())
        
        # Primera fila - 2 gráficos
        col1, col2 = st.columns(2)
        for idx, col in enumerate([col1, col2]):
            if idx < len(letras_list):
                letra = letras_list[idx]
                ticker = mapeo[letra]
                serie = datos[ticker]
                
                with col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=serie.index,  # Usar fechas reales
                        y=serie.values,
                        mode='lines',
                        name=f'Activo {letra}',
                        line=dict(width=2.5, color='#1f77b4')
                    ))
                    fig.update_layout(
                        title=dict(text=f"<b>Activo {letra}</b>", font=dict(size=16)),
                        height=350,
                        showlegend=False,
                        margin=dict(l=50, r=30, t=50, b=50),
                        xaxis_title="Fecha",
                        yaxis_title="Precio (USD)",
                        hovermode='x',
                        plot_bgcolor='rgba(240,240,240,0.5)',
                        xaxis=dict(tickformat='%Y-%m-%d')
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Segunda fila - 2 gráficos
        if len(letras_list) > 2:
            col3, col4 = st.columns(2)
            for idx, col in enumerate([col3, col4], start=2):
                if idx < len(letras_list):
                    letra = letras_list[idx]
                    ticker = mapeo[letra]
                    serie = datos[ticker]
                    
                    with col:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=serie.index,  # Usar fechas reales
                            y=serie.values,
                            mode='lines',
                            name=f'Activo {letra}',
                            line=dict(width=2.5, color='#1f77b4')
                        ))
                        fig.update_layout(
                            title=dict(text=f"<b>Activo {letra}</b>", font=dict(size=16)),
                            height=350,
                            showlegend=False,
                            margin=dict(l=50, r=30, t=50, b=50),
                            xaxis_title="Fecha",
                            yaxis_title="Precio (USD)",
                            hovermode='x',
                            plot_bgcolor='rgba(240,240,240,0.5)',
                            xaxis=dict(tickformat='%Y-%m-%d')
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Input del estudiante
        st.markdown("**Ordena los activos de menor a mayor riesgo:**")
        orden_estudiante = st.multiselect(
            "Selecciona en orden (primero el menos riesgoso, último el más riesgoso):",
            options=list(mapeo.keys()),
            key="orden_volatilidad"
        )
        
        if st.button("✅ Verificar Orden", key="btn_verificar_volatilidad"):
            if len(orden_estudiante) == len(datos):
                # Calcular orden correcto basado en volatilidades
                orden_correcto = sorted(mapeo.keys(), key=lambda x: volatilidades[mapeo[x]])
                
                # Verificar respuesta
                if orden_estudiante == orden_correcto:
                    st.success("🎉 ¡Excelente! Has ordenado correctamente los activos por volatilidad.")
                    st.session_state.puntos_mercado += 10
                    st.balloons()
                else:
                    st.error("❌ El orden no es correcto. Intenta nuevamente.")
                    st.session_state.puntos_mercado += 2
                
                # Mostrar orden correcto y volatilidades
                st.markdown("### 📊 Orden Correcto:")
                for letra in orden_correcto:
                    ticker = mapeo[letra]
                    st.write(f"**Activo {letra}** ({ticker}): Volatilidad anualizada = {volatilidades[ticker]:.2f}%")
                
                # Botón para reiniciar juego
                if st.button("🔄 Nuevo Juego", key="btn_reset_volatilidad"):
                    if 'tickers_juego1' in st.session_state:
                        del st.session_state.tickers_juego1
                    if 'mapeo_activos' in st.session_state:
                        del st.session_state.mapeo_activos
                    st.rerun()
            else:
                st.warning("⚠️ Por favor selecciona todos los activos en el orden correcto.")
    
    st.divider()
    
    # ========================================================================
    # JUEGO 2: SHOCK DE MERCADO EN PORTAFOLIO
    # ========================================================================
    
    st.subheader("💥 Juego 2: Shock de Mercado en Portafolio")
    st.markdown("""
    Construye un portafolio con 3 activos y simula cómo afecta una caída del mercado. 
    Observa la diferencia entre concentración y diversificación.
    """)
    
    st.markdown("### 🎯 Paso 1: Construye tu Portafolio")
    
    # Seleccionar 3 activos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        activo1 = st.selectbox("Activo 1:", tickers, key="activo1_shock", index=0)
        if activo1 in volatilidades_globales:
            st.info(f"📊 **Volatilidad:** {volatilidades_globales[activo1]:.2f}%")
        else:
            st.info("📊 **Volatilidad:** N/A")
        peso1 = st.slider("Peso % Activo 1:", 0, 100, 33, 5, key="peso1_shock")
    
    with col2:
        activo2 = st.selectbox("Activo 2:", tickers, key="activo2_shock", index=1)
        if activo2 in volatilidades_globales:
            st.info(f"📊 **Volatilidad:** {volatilidades_globales[activo2]:.2f}%")
        else:
            st.info("📊 **Volatilidad:** N/A")
        peso2 = st.slider("Peso % Activo 2:", 0, 100, 33, 5, key="peso2_shock")
    
    with col3:
        activo3 = st.selectbox("Activo 3:", tickers, key="activo3_shock", index=2)
        if activo3 in volatilidades_globales:
            st.info(f"📊 **Volatilidad:** {volatilidades_globales[activo3]:.2f}%")
        else:
            st.info("📊 **Volatilidad:** N/A")
        peso3 = st.slider("Peso % Activo 3:", 0, 100, 34, 5, key="peso3_shock")
    
    # Validar que los pesos sumen 100%
    peso_total = peso1 + peso2 + peso3
    
    if peso_total != 100:
        st.warning(f"⚠️ Los pesos deben sumar 100%. Actualmente suman {peso_total}%")
    else:
        st.success(f"✅ Portafolio válido: {peso1}% {activo1} + {peso2}% {activo2} + {peso3}% {activo3}")
        
        # Calcular volatilidad ponderada del portafolio
        activos_port = [activo1, activo2, activo3]
        pesos_port = [peso1/100, peso2/100, peso3/100]
        
        volatilidad_portafolio = 0
        vol_disponibles = []
        
        for activo, peso in zip(activos_port, pesos_port):
            if activo in volatilidades_globales:
                volatilidad_portafolio += volatilidades_globales[activo] * peso
                vol_disponibles.append(volatilidades_globales[activo])
        
        if len(vol_disponibles) > 0:
            vol_max = max(vol_disponibles)
            vol_min = min(vol_disponibles)
            
            st.info(f"""
            📊 **Estadísticas del Portafolio:**
            - **Volatilidad ponderada:** {volatilidad_portafolio:.2f}% anualizada
            - **Activo más volátil:** {vol_max:.2f}%
            - **Activo menos volátil:** {vol_min:.2f}%
            - **Rango de volatilidad:** {vol_max - vol_min:.2f}%
            
            💡 *La diversificación puede reducir el riesgo si los activos tienen volatilidades diferentes.*
            """)
    
    st.markdown("### 💰 Paso 2: Define tu Inversión y el Shock")
    
    col1, col2 = st.columns(2)
    
    with col1:
        inversion_inicial = st.number_input("Inversión inicial (USD):", min_value=1000, max_value=1000000, value=10000, step=1000, key="inversion_portafolio")
    
    with col2:
        caida_porcentaje = st.selectbox("Caída simulada del mercado:", ["-3%", "-5%", "-10%", "-15%", "-20%", "-25%"], key="caida_portafolio")
    
    if st.button("🎲 Simular Shock en Portafolio", key="btn_shock_portafolio") and peso_total == 100:
        caida = float(caida_porcentaje.strip('%')) / 100
        
        st.markdown("### 📊 Resultados del Shock")
        
        # Crear DataFrame con la información del portafolio
        activos_seleccionados = [activo1, activo2, activo3]
        pesos = [peso1/100, peso2/100, peso3/100]
        inversiones = [inversion_inicial * p for p in pesos]
        
        # Simular diferentes impactos por activo (basado en volatilidad si está disponible)
        impactos = []
        valores_finales = []
        perdidas = []
        
        for i, activo in enumerate(activos_seleccionados):
            # Si tenemos volatilidad, ajustar el impacto proporcionalmente
            if activo in volatilidades_globales and len(volatilidades_globales) > 0:
                vol_promedio = sum(volatilidades_globales.values()) / len(volatilidades_globales)
                factor_ajuste = volatilidades_globales[activo] / vol_promedio if vol_promedio > 0 else 1.0
                impacto_activo = caida * factor_ajuste
            else:
                impacto_activo = caida
            
            impactos.append(impacto_activo)
            valor_final = inversiones[i] * (1 + impacto_activo)
            valores_finales.append(valor_final)
            perdidas.append(inversiones[i] - valor_final)
        
        # Calcular totales del portafolio
        valor_final_portafolio = sum(valores_finales)
        perdida_total = inversion_inicial - valor_final_portafolio
        retorno_portafolio = (valor_final_portafolio - inversion_inicial) / inversion_inicial * 100
        
        # Mostrar métricas principales
        col1, col2, col3 = st.columns(3)
        col1.metric("Inversión Inicial", f"${inversion_inicial:,.2f}")
        col2.metric("Valor Después del Shock", f"${valor_final_portafolio:,.2f}", f"{retorno_portafolio:.2f}%")
        col3.metric("Pérdida Total", f"${perdida_total:,.2f}")
        
        # Tabla detallada por activo
        st.markdown("#### 📋 Detalle por Activo")
        
        import pandas as pd
        df_resultados = pd.DataFrame({
            'Activo': activos_seleccionados,
            'Peso (%)': [f"{p*100:.1f}%" for p in pesos],
            'Inversión Inicial': [f"${inv:,.2f}" for inv in inversiones],
            'Impacto (%)': [f"{imp*100:.2f}%" for imp in impactos],
            'Valor Final': [f"${vf:,.2f}" for vf in valores_finales],
            'Pérdida': [f"${p:,.2f}" for p in perdidas]
        })
        
        st.dataframe(df_resultados, use_container_width=True)
        
        # Visualización: Gráfico de torta del portafolio
        col1, col2 = st.columns(2)
        
        with col1:
            fig_inicial = go.Figure(data=[go.Pie(
                labels=[f"{a} ({p*100:.0f}%)" for a, p in zip(activos_seleccionados, pesos)],
                values=inversiones,
                hole=0.4,
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
            )])
            fig_inicial.update_layout(
                title="<b>Portafolio Inicial</b>",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_inicial, use_container_width=True)
        
        with col2:
            fig_final = go.Figure(data=[go.Pie(
                labels=[f"{a} ({p*100:.0f}%)" for a, p in zip(activos_seleccionados, pesos)],
                values=valores_finales,
                hole=0.4,
                marker=dict(colors=['#d62728', '#ff7f0e', '#2ca02c'])
            )])
            fig_final.update_layout(
                title="<b>Portafolio Después del Shock</b>",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_final, use_container_width=True)
        
        # Gráfico de barras comparativo
        fig_barras = go.Figure()
        
        fig_barras.add_trace(go.Bar(
            name='Inversión Inicial',
            x=activos_seleccionados,
            y=inversiones,
            marker_color='#1f77b4'
        ))
        
        fig_barras.add_trace(go.Bar(
            name='Valor Después del Shock',
            x=activos_seleccionados,
            y=valores_finales,
            marker_color='#d62728'
        ))
        
        fig_barras.update_layout(
            title="<b>Impacto por Activo</b>",
            xaxis_title="Activo",
            yaxis_title="Valor (USD)",
            barmode='group',
            height=350
        )
        
        st.plotly_chart(fig_barras, use_container_width=True)
        
        # Mensaje educativo
        st.info(f"""
        📚 **Lección de Diversificación:**
        
        Tu portafolio está compuesto por:
        - **{activo1}** ({peso1}%): Pérdida de ${perdidas[0]:,.2f}
        - **{activo2}** ({peso2}%): Pérdida de ${perdidas[1]:,.2f}
        - **{activo3}** ({peso3}%): Pérdida de ${perdidas[2]:,.2f}
        
        **Pérdida total del portafolio:** ${perdida_total:,.2f} ({retorno_portafolio:.2f}%)
        
        💡 **Observación:** Los activos con mayor volatilidad histórica tienden a experimentar 
        caídas más pronunciadas durante shocks de mercado. Un portafolio diversificado puede 
        ayudar a mitigar el impacto cuando los activos no se mueven en perfecta sincronía.
        
        🎯 **Estrategia:** Considera balancear activos de diferentes clases (acciones, bonos, 
        materias primas) y sectores para reducir la correlación y el riesgo total del portafolio.
        """)
        
        st.session_state.puntos_mercado += 10
    
    # Mostrar puntos de esta pestaña
    st.divider()
    st.success(f"🎯 Puntos en Riesgo de Mercado: {st.session_state.puntos_mercado}")


# ============================================================================
# PESTAÑA 2: RIESGO FINANCIERO (ALTMAN Z-SCORE)
# ============================================================================

def tab_riesgo_financiero():
    st.header("💼 Riesgo Financiero - Modelo Altman Z-Score")
    st.markdown("""
    El **Z-Score de Altman** es un modelo que predice la probabilidad de quiebra de una empresa 
    basándose en ratios financieros. Desarrollado por Edward Altman, es ampliamente utilizado 
    para evaluar el riesgo crediticio.
    
    **Fórmula:** Z = 0.717×X1 + 0.847×X2 + 3.107×X3 + 0.420×X4 + 0.998×X5
    
    Donde:
    - **X1** = Capital de trabajo / Total de activos (Liquidez)
    - **X2** = Utilidades retenidas / Total de activos (Historial de rentabilidad)
    - **X3** = EBIT / Total de activos (Rentabilidad operativa)
    - **X4** = Valor de mercado del patrimonio / Total de pasivos (Apalancamiento)
    - **X5** = Ventas / Total de activos (Rotación de activos)
    """)
    
    st.divider()
    
    # Entrada de ratios
    st.subheader("📝 Paso 1: Ingresa los Ratios Financieros")
    st.markdown("*Nota: Estos valores deben calcularse previamente en Excel desde los estados financieros.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x1 = st.number_input("X1: Capital de trabajo / Total activos", 
                            min_value=-1.0, max_value=1.0, value=0.15, step=0.01,
                            help="Mide la liquidez. Valores típicos: 0.10 a 0.30")
        x2 = st.number_input("X2: Utilidades retenidas / Total activos", 
                            min_value=-1.0, max_value=1.0, value=0.20, step=0.01,
                            help="Mide el historial de rentabilidad. Valores típicos: 0.10 a 0.40")
        x3 = st.number_input("X3: EBIT / Total activos", 
                            min_value=-1.0, max_value=1.0, value=0.10, step=0.01,
                            help="Mide la rentabilidad operativa. Valores típicos: 0.05 a 0.20")
    
    with col2:
        x4 = st.number_input("X4: Valor mercado patrimonio / Total pasivos", 
                            min_value=0.0, max_value=10.0, value=1.5, step=0.1,
                            help="Mide el apalancamiento. Valores típicos: 0.50 a 3.00")
        x5 = st.number_input("X5: Ventas / Total activos", 
                            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                            help="Mide la eficiencia de activos. Valores típicos: 0.80 a 2.00")
    
    if st.button("🧮 Calcular Z-Score", key="btn_calcular_z"):
        # Calcular Z-Score con la fórmula correcta
        z_score = 0.717*x1 + 0.847*x2 + 3.107*x3 + 0.420*x4 + 0.998*x5
        
        # Guardar en session_state
        st.session_state.z_score = z_score
        st.session_state.ratios = {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5}
        
        # Determinar zona según los puntos de corte correctos para Z-Score de Altman
        if z_score >= 2.90:
            zona = "Zona Segura 🟢"
            color = "green"
            probabilidad_quiebra = "Baja"
            mensaje = "La empresa tiene baja probabilidad de quiebra según el modelo. Situación financiera saludable."
        elif z_score >= 1.23:
            zona = "Zona de Alerta 🟡"
            color = "orange"
            probabilidad_quiebra = "Moderada-Alta"
            mensaje = "Zona gris: la empresa no está claramente quebrando, pero tampoco se la puede considerar sana. Se recomienda análisis más profundo, escenarios y stress tests."
        else:
            zona = "Riesgo de Quiebra 🔴"
            color = "red"
            probabilidad_quiebra = "Alta"
            mensaje = "Alta probabilidad de quiebra / problemas financieros serios en el corto plazo. Situación financiera crítica que requiere atención inmediata."
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("📊 Resultados del Análisis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Z-Score", f"{z_score:.2f}")
        with col2:
            st.markdown(f"<h3 style='color: {color};'>{zona}</h3>", unsafe_allow_html=True)
        with col3:
            st.metric("Prob. Quiebra", probabilidad_quiebra)
        
        st.info(mensaje)
        
        # Análisis detallado de cada ratio
        st.markdown("### 🔍 Análisis Detallado por Ratio")
        
        # Evaluar cada ratio individualmente
        analisis_ratios = []
        
        # X1 - Liquidez
        if x1 < 0.10:
            analisis_ratios.append({
                'ratio': 'X1 (Liquidez)',
                'valor': x1,
                'estado': '🔴 CRÍTICO',
                'problema': 'Capital de trabajo muy bajo',
                'recomendacion': 'Mejorar gestión de cobros, reducir inventarios excesivos, renegociar plazos con proveedores'
            })
        elif x1 < 0.20:
            analisis_ratios.append({
                'ratio': 'X1 (Liquidez)',
                'valor': x1,
                'estado': '🟡 MEJORABLE',
                'problema': 'Capital de trabajo ajustado',
                'recomendacion': 'Monitorear flujo de caja, optimizar ciclo de conversión de efectivo'
            })
        else:
            analisis_ratios.append({
                'ratio': 'X1 (Liquidez)',
                'valor': x1,
                'estado': '🟢 ADECUADO',
                'problema': 'N/A',
                'recomendacion': 'Mantener disciplina en gestión de capital de trabajo'
            })
        
        # X2 - Utilidades Retenidas
        if x2 < 0.10:
            analisis_ratios.append({
                'ratio': 'X2 (Utilidades Retenidas)',
                'valor': x2,
                'estado': '🔴 CRÍTICO',
                'problema': 'Historial de pérdidas o utilidades muy bajas',
                'recomendacion': 'Reducir dividendos temporalmente, implementar plan de mejora de rentabilidad, revisar estructura de costos'
            })
        elif x2 < 0.25:
            analisis_ratios.append({
                'ratio': 'X2 (Utilidades Retenidas)',
                'valor': x2,
                'estado': '🟡 MEJORABLE',
                'problema': 'Acumulación de utilidades limitada',
                'recomendacion': 'Balancear política de dividendos, reinvertir utilidades en crecimiento sostenible'
            })
        else:
            analisis_ratios.append({
                'ratio': 'X2 (Utilidades Retenidas)',
                'valor': x2,
                'estado': '🟢 ADECUADO',
                'problema': 'N/A',
                'recomendacion': 'Continuar política de retención de utilidades equilibrada'
            })
        
        # X3 - EBIT / Activos
        if x3 < 0.05:
            analisis_ratios.append({
                'ratio': 'X3 (Rentabilidad Operativa)',
                'valor': x3,
                'estado': '🔴 CRÍTICO',
                'problema': 'Rentabilidad operativa muy baja o negativa',
                'recomendacion': 'Reestructurar operaciones, reducir costos fijos, mejorar márgenes, revisar estrategia de precios'
            })
        elif x3 < 0.10:
            analisis_ratios.append({
                'ratio': 'X3 (Rentabilidad Operativa)',
                'valor': x3,
                'estado': '🟡 MEJORABLE',
                'problema': 'Márgenes operativos ajustados',
                'recomendacion': 'Optimizar eficiencia operativa, buscar economías de escala, mejorar productividad'
            })
        else:
            analisis_ratios.append({
                'ratio': 'X3 (Rentabilidad Operativa)',
                'valor': x3,
                'estado': '🟢 ADECUADO',
                'problema': 'N/A',
                'recomendacion': 'Mantener foco en eficiencia operativa y control de costos'
            })
        
        # X4 - Patrimonio / Pasivos
        if x4 < 0.50:
            analisis_ratios.append({
                'ratio': 'X4 (Estructura de Capital)',
                'valor': x4,
                'estado': '🔴 CRÍTICO',
                'problema': 'Exceso de apalancamiento, patrimonio insuficiente',
                'recomendacion': 'Capitalizar la empresa urgentemente, convertir deuda en equity, reducir pasivos mediante ventas de activos'
            })
        elif x4 < 1.00:
            analisis_ratios.append({
                'ratio': 'X4 (Estructura de Capital)',
                'valor': x4,
                'estado': '🟡 MEJORABLE',
                'problema': 'Apalancamiento elevado',
                'recomendacion': 'Reducir deuda gradualmente, fortalecer patrimonio mediante retención de utilidades'
            })
        else:
            analisis_ratios.append({
                'ratio': 'X4 (Estructura de Capital)',
                'valor': x4,
                'estado': '🟢 ADECUADO',
                'problema': 'N/A',
                'recomendacion': 'Mantener estructura de capital equilibrada'
            })
        
        # X5 - Ventas / Activos (Rotación)
        if x5 < 0.80:
            analisis_ratios.append({
                'ratio': 'X5 (Rotación de Activos)',
                'valor': x5,
                'estado': '🔴 CRÍTICO',
                'problema': 'Baja eficiencia en el uso de activos',
                'recomendacion': 'Optimizar uso de activos, vender activos improductivos, mejorar estrategia comercial, aumentar ventas'
            })
        elif x5 < 1.20:
            analisis_ratios.append({
                'ratio': 'X5 (Rotación de Activos)',
                'valor': x5,
                'estado': '🟡 MEJORABLE',
                'problema': 'Eficiencia de activos moderada',
                'recomendacion': 'Mejorar productividad de activos, revisar mix de productos, optimizar inventarios'
            })
        else:
            analisis_ratios.append({
                'ratio': 'X5 (Rotación de Activos)',
                'valor': x5,
                'estado': '🟢 ADECUADO',
                'problema': 'N/A',
                'recomendacion': 'Mantener eficiencia en rotación de activos'
            })
        
        # Mostrar tabla de análisis
        import pandas as pd
        df_analisis = pd.DataFrame(analisis_ratios)
        st.dataframe(df_analisis, use_container_width=True, hide_index=True)
        
        # Recomendaciones prioritarias
        st.markdown("### 🎯 Plan de Acción Prioritario")
        
        ratios_criticos = [r for r in analisis_ratios if '🔴' in r['estado']]
        ratios_mejorables = [r for r in analisis_ratios if '🟡' in r['estado']]
        
        if len(ratios_criticos) > 0:
            st.error("**⚠️ ATENCIÓN INMEDIATA REQUERIDA:**")
            for i, ratio in enumerate(ratios_criticos, 1):
                st.markdown(f"**{i}. {ratio['ratio']}** ({ratio['valor']:.3f})")
                st.markdown(f"   - **Problema:** {ratio['problema']}")
                st.markdown(f"   - **Acción:** {ratio['recomendacion']}")
                st.markdown("")
        
        if len(ratios_mejorables) > 0:
            st.warning("**📋 ACCIONES DE MEJORA:**")
            for i, ratio in enumerate(ratios_mejorables, 1):
                st.markdown(f"**{i}. {ratio['ratio']}** ({ratio['valor']:.3f})")
                st.markdown(f"   - **Observación:** {ratio['problema']}")
                st.markdown(f"   - **Recomendación:** {ratio['recomendacion']}")
                st.markdown("")
        
        if len(ratios_criticos) == 0 and len(ratios_mejorables) == 0:
            st.success("**✅ EMPRESA SALUDABLE:**")
            st.markdown("""
            Todos los ratios están en rangos adecuados. Recomendaciones generales:
            - Mantener disciplina financiera
            - Monitorear cambios en el entorno competitivo
            - Seguir optimizando eficiencia operativa
            - Diversificar fuentes de ingresos
            """)
        
        # Mostrar gráfico de contribución de cada ratio
        contribuciones = {
            'X1 (Liquidez)': 0.717*x1,
            'X2 (Util. Retenidas)': 0.847*x2,
            'X3 (EBIT)': 3.107*x3,
            'X4 (Valor Mercado/Pasivo)': 0.420*x4,
            'X5 (Ventas/Activos)': 0.998*x5
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(contribuciones.keys()), y=list(contribuciones.values()),
                  marker_color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink'])
        ])
        fig.update_layout(
            title="Contribución de cada componente al Z-Score",
            xaxis_title="Componente",
            yaxis_title="Contribución",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.puntos_financiero += 5
    
    st.divider()
    
    # ========================================================================
    # JUEGO: ENCUENTRA EL RATIO PROBLEMÁTICO
    # ========================================================================
    
    if 'z_score' in st.session_state and 'ratios' in st.session_state:
        st.subheader("🎯 Juego 1: Encuentra el Ratio Problemático")
        st.markdown("Identifica cuál es el ratio más crítico que está afectando negativamente el Z'-Score de la empresa.")
        
        ratios = st.session_state.ratios
        
        # Normalizar ratios para comparación (X4 y X5 tienen escalas diferentes)
        ratios_normalizados = {
            'X1': ratios['X1'],
            'X2': ratios['X2'],
            'X3': ratios['X3'],
            'X4': ratios['X4'] / 2,  # Normalizar X4
            'X5': ratios['X5'] / 2   # Normalizar X5
        }
        
        # Encontrar el ratio más bajo
        ratio_critico = min(ratios_normalizados, key=ratios_normalizados.get)
        
        # Pregunta al estudiante
        respuesta_estudiante = st.radio(
            "¿Cuál crees que es el ratio más problemático para esta empresa?",
            options=[
                "Liquidez (X1: Capital de trabajo / Activos)",
                "Utilidades retenidas / Historial (X2: Utilidades retenidas / Activos)",
                "Rentabilidad operativa (X3: EBIT / Activos)",
                "Apalancamiento / Estructura de capital (X4: Valor mercado patrimonio / Pasivos)",
                "Rotación de activos (X5: Ventas / Activos)"
            ],
            key="radio_ratio_critico"
        )
        
        mapeo_respuestas = {
            "Liquidez (X1: Capital de trabajo / Activos)": 'X1',
            "Utilidades retenidas / Historial (X2: Utilidades retenidas / Activos)": 'X2',
            "Rentabilidad operativa (X3: EBIT / Activos)": 'X3',
            "Apalancamiento / Estructura de capital (X4: Valor mercado patrimonio / Pasivos)": 'X4',
            "Rotación de activos (X5: Ventas / Activos)": 'X5'
        }
        
        if st.button("✅ Verificar Respuesta", key="btn_verificar_ratio"):
            ratio_elegido = mapeo_respuestas[respuesta_estudiante]
            
            if ratio_elegido == ratio_critico:
                st.success(f"🎉 ¡Correcto! El ratio {ratio_critico} es el más problemático con un valor de {ratios[ratio_critico]:.3f}")
                st.session_state.puntos_financiero += 10
                st.balloons()
            else:
                st.error(f"❌ Incorrecto. El ratio más problemático es {ratio_critico} con un valor de {ratios[ratio_critico]:.3f}")
                st.session_state.puntos_financiero += 3
            
            # Guardar ratio crítico para el siguiente juego
            st.session_state.ratio_critico = ratio_critico
            
            # Análisis detallado
            st.markdown("### 📈 Análisis de Ratios:")
            for ratio, valor in ratios.items():
                if ratio == ratio_critico:
                    st.warning(f"**{ratio}**: {valor:.3f} ⚠️ (Ratio crítico)")
                else:
                    st.write(f"**{ratio}**: {valor:.3f}")
        
        st.divider()
        
        # ====================================================================
        # JUEGO: PROPÓN LA SOLUCIÓN CORRECTA
        # ====================================================================
        
        if 'ratio_critico' in st.session_state:
            st.subheader("🎯 Juego 2: Propón la Solución Correcta")
            st.markdown(f"El ratio crítico identificado es **{st.session_state.ratio_critico}**. ¿Cuál es la mejor estrategia para mejorarlo?")
            
            ratio_critico = st.session_state.ratio_critico
            
            # Definir opciones según el ratio crítico
            if ratio_critico == 'X1':  # Problema de liquidez
                st.info("**Problema:** Capital de trabajo insuficiente (baja liquidez)")
                opciones = [
                    "Reducir inventarios y mejorar la gestión de cobros para liberar efectivo",
                    "Refinanciar deuda de corto plazo a largo plazo",
                    "Aumentar agresivamente la deuda de corto plazo",
                    "Vender activos fijos estratégicos"
                ]
                solucion_correcta = opciones[0]
                explicacion_correcta = "Reducir inventarios y mejorar cobros aumenta el capital de trabajo sin comprometer la operación."
                explicacion_incorrecta = {
                    opciones[1]: "Aunque ayuda, no soluciona el problema de fondo de liquidez operativa.",
                    opciones[2]: "Esto empeoraría el problema al aumentar pasivos corrientes.",
                    opciones[3]: "Vender activos estratégicos puede comprometer la operación futura."
                }
            
            elif ratio_critico == 'X2':  # Problema de utilidades retenidas
                st.info("**Problema:** Bajo historial de rentabilidad acumulada")
                opciones = [
                    "Retener más utilidades y reducir dividendos temporalmente",
                    "Aumentar dividendos para atraer inversores",
                    "Tomar más deuda para financiar operaciones",
                    "Vender activos no rentables"
                ]
                solucion_correcta = opciones[0]
                explicacion_correcta = "Retener utilidades aumenta directamente este ratio y fortalece el patrimonio."
                explicacion_incorrecta = {
                    opciones[1]: "Esto reduciría aún más las utilidades retenidas.",
                    opciones[2]: "La deuda no afecta las utilidades retenidas directamente.",
                    opciones[3]: "Puede ayudar, pero no mejora el historial de rentabilidad."
                }
            
            elif ratio_critico == 'X3':  # Problema de rentabilidad operativa
                st.info("**Problema:** Baja rentabilidad operativa (EBIT bajo)")
                opciones = [
                    "Reducir costos operativos y mejorar eficiencia",
                    "Tomar más deuda para invertir en marketing",
                    "Reducir precios para aumentar volumen de ventas",
                    "Distribuir más dividendos"
                ]
                solucion_correcta = opciones[0]
                explicacion_correcta = "Reducir costos mejora el EBIT directamente sin afectar ingresos."
                explicacion_incorrecta = {
                    opciones[1]: "Más deuda aumenta gastos financieros y puede reducir el EBIT.",
                    opciones[2]: "Reducir precios puede disminuir el margen y empeorar el EBIT.",
                    opciones[3]: "Los dividendos no afectan el EBIT."
                }
            
            elif ratio_critico == 'X4':  # X4 - Problema de apalancamiento
                st.info("**Problema:** Excesivo apalancamiento (bajo patrimonio vs pasivos)")
                opciones = [
                    "Emitir nuevas acciones para aumentar el patrimonio",
                    "Tomar más deuda para financiar expansión",
                    "Aumentar dividendos",
                    "Reducir el capital social"
                ]
                solucion_correcta = opciones[0]
                explicacion_correcta = "Emitir acciones aumenta el patrimonio y mejora el ratio X4 directamente."
                explicacion_incorrecta = {
                    opciones[1]: "Más deuda empeoraría el apalancamiento.",
                    opciones[2]: "Los dividendos reducen el patrimonio.",
                    opciones[3]: "Esto empeoraría el problema al reducir patrimonio."
                }
            
            else:  # X5 - Problema de rotación de activos
                st.info("**Problema:** Baja rotación de activos (ventas insuficientes vs activos)")
                opciones = [
                    "Aumentar ventas mediante marketing y expansión comercial",
                    "Vender activos improductivos o subutilizados",
                    "Comprar más activos fijos para expandir capacidad",
                    "Reducir precios drásticamente sin análisis de rentabilidad"
                ]
                solucion_correcta = opciones[0]
                explicacion_correcta = "Aumentar ventas mejora directamente el ratio X5 (Ventas/Activos) sin comprometer la base de activos."
                explicacion_incorrecta = {
                    opciones[1]: "Puede ayudar, pero es mejor aumentar ventas primero antes de reducir capacidad.",
                    opciones[2]: "Más activos empeorarían el ratio al aumentar el denominador sin garantizar ventas proporcionales.",
                    opciones[3]: "Reducir precios sin estrategia puede afectar márgenes y rentabilidad (X3)."
                }
            
            # Pregunta al estudiante
            solucion_estudiante = st.radio(
                "Selecciona la mejor estrategia:",
                options=opciones,
                key="radio_solucion"
            )
            
            if st.button("✅ Verificar Solución", key="btn_verificar_solucion"):
                if solucion_estudiante == solucion_correcta:
                    st.success(f"🎉 ¡Excelente decisión! {explicacion_correcta}")
                    st.session_state.puntos_financiero += 15
                    st.balloons()
                else:
                    st.error(f"❌ No es la mejor opción. {explicacion_incorrecta[solucion_estudiante]}")
                    st.write(f"💡 **Mejor solución:** {solucion_correcta}")
                    st.write(f"**Por qué:** {explicacion_correcta}")
                    st.session_state.puntos_financiero += 5
        
        st.divider()
        
        # Resumen final
        st.subheader("📋 Resumen del Análisis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Z-Score", f"{st.session_state.z_score:.2f}")
        with col2:
            if 'ratio_critico' in st.session_state:
                st.metric("Ratio Crítico", st.session_state.ratio_critico)
        with col3:
            st.metric("Puntos Ganados", st.session_state.puntos_financiero)


# ============================================================================
# PESTAÑA 3: RIESGO MACROECONÓMICO
# ============================================================================

def tab_riesgo_macro():
    st.header("📈 Riesgo Macroeconómico")
    st.markdown("""
    El **riesgo macroeconómico** se refiere a cómo los factores económicos generales (inflación, 
    tasas de interés, crecimiento económico) afectan el desempeño de las empresas y las decisiones 
    de inversión.
    """)
    
    st.divider()
    
    # Cargar datos automáticamente desde el archivo
    try:
        archivo_path = "Variables Macroeconómicas.xlsx"
        df = pd.read_excel(archivo_path)
        
        # Detectar columna de fecha (primer columna o columna con 'fecha' en el nombre)
        columna_fecha = None
        for col in df.columns:
            if 'fecha' in col.lower() or 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
                columna_fecha = col
                break
        
        if columna_fecha is None:
            columna_fecha = df.columns[0]  # Usar primera columna
        
        # Convertir a datetime
        df['fecha'] = pd.to_datetime(df[columna_fecha])
        df = df.sort_values('fecha').reset_index(drop=True)
        
        # Identificar columnas numéricas (variables macroeconómicas)
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columnas_numericas) == 0:
            st.error("❌ No se encontraron variables numéricas en el archivo.")
            return
        
        st.success(f"✅ Datos cargados: {len(df)} registros desde {df['fecha'].min().strftime('%Y-%m')} hasta {df['fecha'].max().strftime('%Y-%m')}")
        st.info(f"📊 Variables disponibles: {', '.join(columnas_numericas)}")
        
        # ====================================================================
        # SECCIÓN 1: VISUALIZACIÓN DE VARIABLES
        # ====================================================================
        
        st.subheader("📊 Visualización de Variables Macroeconómicas")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            variable_seleccionada = st.selectbox(
                "Selecciona la variable a graficar:",
                options=columnas_numericas,
                key="var_graficar"
            )
        
        with col2:
            mostrar_todas = st.checkbox("Mostrar todas las variables", value=False, key="check_todas")
        
        if mostrar_todas:
            # Graficar todas las variables
            fig = go.Figure()
            for col in columnas_numericas:
                fig.add_trace(go.Scatter(
                    x=df['fecha'],
                    y=df[col],
                    mode='lines',
                    name=col
                ))
            fig.update_layout(
                title='Todas las Variables Macroeconómicas',
                xaxis_title='Fecha',
                yaxis_title='Valor',
                height=500,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Graficar variable seleccionada
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['fecha'],
                y=df[variable_seleccionada],
                mode='lines+markers',
                name=variable_seleccionada,
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            fig.update_layout(
                title=f'Serie Temporal: {variable_seleccionada}',
                xaxis_title='Fecha',
                yaxis_title=variable_seleccionada,
                height=450,
                hovermode='x'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas descriptivas
            with st.expander("📊 Estadísticas Descriptivas"):
                col1, col2, col3, col4 = st.columns(4)
                serie = df[variable_seleccionada]
                col1.metric("Media", f"{serie.mean():.2f}")
                col2.metric("Mediana", f"{serie.median():.2f}")
                col3.metric("Desv. Estándar", f"{serie.std():.2f}")
                col4.metric("Coef. Variación", f"{(serie.std()/serie.mean()*100):.2f}%")
        
        st.divider()
        
        # ====================================================================
        # SECCIÓN 2: PRONÓSTICOS CON AUTO-ARIMA Y ETS
        # ====================================================================
        
        st.subheader("🔮 Pronósticos con Modelos de Series de Tiempo")
        st.markdown("""
        Compara dos modelos de pronóstico y elige el mejor según sus métricas de precisión:
        - **Auto-ARIMA:** Selecciona automáticamente los mejores parámetros (p,d,q)
        - **ETS (Error, Trend, Seasonality):** Suavización exponencial con componentes aditivos/multiplicativos
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            variable_pronostico = st.selectbox(
                "Variable a pronosticar:",
                options=columnas_numericas,
                key="var_pronostico"
            )
        
        with col2:
            horizonte = st.slider(
                "Horizonte de pronóstico (meses):",
                min_value=3,
                max_value=24,
                value=12,
                step=1,
                key="horizonte"
            )
        
        # Preparar datos para el pronóstico
        serie_pronostico = df[[variable_pronostico]].copy()
        serie_pronostico.index = df['fecha']
        serie_pronostico = serie_pronostico[variable_pronostico]
        
        # Dividir en train/test (últimos 12 meses para validación)
        n_test = min(12, len(serie_pronostico) // 4)
        train = serie_pronostico[:-n_test]
        test = serie_pronostico[-n_test:]
        
        col1, col2 = st.columns(2)
        
        # ============================================================
        # MODELO 1: AUTO-ARIMA
        # ============================================================
        
        with col1:
            if st.button("📈 Ajustar Auto-ARIMA", key="btn_autoarima"):
                with st.spinner("Ajustando modelo Auto-ARIMA..."):
                    try:
                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                        from pmdarima import auto_arima
                        
                        # Auto-ARIMA para encontrar mejores parámetros
                        modelo_auto = auto_arima(
                            train,
                            seasonal=False,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            max_p=5,
                            max_q=5,
                            max_d=2
                        )
                        
                        # Obtener orden óptimo
                        orden = modelo_auto.order
                        
                        # Ajustar SARIMAX con serie completa
                        modelo_final = SARIMAX(serie_pronostico, order=orden)
                        resultado = modelo_final.fit(disp=False)
                        
                        # Pronóstico
                        pronostico_arima = resultado.forecast(steps=horizonte)
                        
                        # Validación en test set
                        pred_test = resultado.predict(start=len(train), end=len(serie_pronostico)-1)
                        rmse_arima = np.sqrt(np.mean((test - pred_test)**2))
                        aic_arima = resultado.aic
                        bic_arima = resultado.bic
                        
                        # Guardar en session_state
                        st.session_state.arima_resultado = {
                            'pronostico': pronostico_arima,
                            'rmse': rmse_arima,
                            'aic': aic_arima,
                            'bic': bic_arima,
                            'orden': orden
                        }
                        
                        # Graficar
                        fig = go.Figure()
                        
                        # Histórico
                        fig.add_trace(go.Scatter(
                            x=serie_pronostico.index,
                            y=serie_pronostico.values,
                            mode='lines',
                            name='Histórico',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Pronóstico
                        fechas_futuras = pd.date_range(
                            start=serie_pronostico.index[-1] + pd.DateOffset(months=1),
                            periods=horizonte,
                            freq='MS'
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=fechas_futuras,
                            y=pronostico_arima,
                            mode='lines+markers',
                            name='Pronóstico ARIMA',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f'Auto-ARIMA{orden} - {variable_pronostico}',
                            xaxis_title='Fecha',
                            yaxis_title=variable_pronostico,
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Métricas
                        st.success("✅ Modelo Auto-ARIMA ajustado exitosamente")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("RMSE", f"{rmse_arima:.4f}")
                        col_b.metric("AIC", f"{aic_arima:.2f}")
                        col_c.metric("BIC", f"{bic_arima:.2f}")
                        
                        st.info(f"📊 **Orden seleccionado:** ARIMA{orden}")
                        
                        # Tabla de pronósticos
                        with st.expander("📋 Ver valores pronosticados"):
                            df_forecast = pd.DataFrame({
                                'Fecha': fechas_futuras,
                                'Pronóstico': pronostico_arima.round(4)
                            })
                            st.dataframe(df_forecast, use_container_width=True, hide_index=True)
                        
                        st.session_state.puntos_macro += 10
                        
                    except Exception as e:
                        st.error(f"❌ Error al ajustar Auto-ARIMA: {str(e)}")
                        st.info("💡 Intenta con una serie más larga o verifica que no haya valores faltantes.")
        
        # ============================================================
        # MODELO 2: ETS
        # ============================================================
        
        with col2:
            if st.button("📊 Ajustar ETS", key="btn_ets"):
                with st.spinner("Ajustando modelo ETS..."):
                    try:
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing
                        
                        # Ajustar modelo ETS
                        modelo_ets = ExponentialSmoothing(
                            serie_pronostico,
                            trend='add',
                            seasonal='add' if len(serie_pronostico) >= 24 else None,
                            seasonal_periods=12 if len(serie_pronostico) >= 24 else None
                        )
                        resultado_ets = modelo_ets.fit()
                        
                        # Pronóstico
                        pronostico_ets = resultado_ets.forecast(steps=horizonte)
                        
                        # Validación en test set
                        pred_test_ets = resultado_ets.predict(start=len(train), end=len(serie_pronostico)-1)
                        rmse_ets = np.sqrt(np.mean((test - pred_test_ets)**2))
                        aic_ets = resultado_ets.aic
                        bic_ets = resultado_ets.bic
                        
                        # Guardar en session_state
                        st.session_state.ets_resultado = {
                            'pronostico': pronostico_ets,
                            'rmse': rmse_ets,
                            'aic': aic_ets,
                            'bic': bic_ets
                        }
                        
                        # Graficar
                        fig = go.Figure()
                        
                        # Histórico
                        fig.add_trace(go.Scatter(
                            x=serie_pronostico.index,
                            y=serie_pronostico.values,
                            mode='lines',
                            name='Histórico',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Pronóstico
                        fechas_futuras = pd.date_range(
                            start=serie_pronostico.index[-1] + pd.DateOffset(months=1),
                            periods=horizonte,
                            freq='MS'
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=fechas_futuras,
                            y=pronostico_ets,
                            mode='lines+markers',
                            name='Pronóstico ETS',
                            line=dict(color='green', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f'ETS - {variable_pronostico}',
                            xaxis_title='Fecha',
                            yaxis_title=variable_pronostico,
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Métricas
                        st.success("✅ Modelo ETS ajustado exitosamente")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("RMSE", f"{rmse_ets:.4f}")
                        col_b.metric("AIC", f"{aic_ets:.2f}")
                        col_c.metric("BIC", f"{bic_ets:.2f}")
                        
                        # Tabla de pronósticos
                        with st.expander("📋 Ver valores pronosticados"):
                            df_forecast = pd.DataFrame({
                                'Fecha': fechas_futuras,
                                'Pronóstico': pronostico_ets.round(4)
                            })
                            st.dataframe(df_forecast, use_container_width=True, hide_index=True)
                        
                        st.session_state.puntos_macro += 10
                        
                    except Exception as e:
                        st.error(f"❌ Error al ajustar ETS: {str(e)}")
                        st.info("💡 El modelo ETS requiere al menos 2 años de datos para capturar estacionalidad.")
        
        # ============================================================
        # COMPARACIÓN DE MODELOS
        # ============================================================
        
        if 'arima_resultado' in st.session_state and 'ets_resultado' in st.session_state:
            st.divider()
            st.subheader("🏆 Comparación de Modelos")
            
            arima_res = st.session_state.arima_resultado
            ets_res = st.session_state.ets_resultado
            
            # Tabla comparativa
            df_comparacion = pd.DataFrame({
                'Modelo': ['Auto-ARIMA', 'ETS'],
                'RMSE': [arima_res['rmse'], ets_res['rmse']],
                'AIC': [arima_res['aic'], ets_res['aic']],
                'BIC': [arima_res['bic'], ets_res['bic']]
            })
            
            st.dataframe(df_comparacion, use_container_width=True, hide_index=True)
            
            # Determinar mejor modelo
            mejor_rmse = 'Auto-ARIMA' if arima_res['rmse'] < ets_res['rmse'] else 'ETS'
            mejor_aic = 'Auto-ARIMA' if arima_res['aic'] < ets_res['aic'] else 'ETS'
            mejor_bic = 'Auto-ARIMA' if arima_res['bic'] < ets_res['bic'] else 'ETS'
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mejor RMSE", mejor_rmse, "↓" if mejor_rmse == 'Auto-ARIMA' else "↑")
            col2.metric("Mejor AIC", mejor_aic, "↓" if mejor_aic == 'Auto-ARIMA' else "↑")
            col3.metric("Mejor BIC", mejor_bic, "↓" if mejor_bic == 'Auto-ARIMA' else "↑")
            
            st.info("""
            📚 **Guía de Selección:**
            - **RMSE (Root Mean Square Error):** Mide el error de predicción. Menor es mejor.
            - **AIC (Akaike Information Criterion):** Balance entre ajuste y complejidad. Menor es mejor.
            - **BIC (Bayesian Information Criterion):** Similar al AIC pero penaliza más la complejidad. Menor es mejor.
            
            💡 **Recomendación:** Si los tres criterios favorecen el mismo modelo, ese es tu mejor opción. 
            Si hay discrepancias, prioriza el RMSE para precisión de pronóstico.
            """)
            
            # Pregunta al estudiante
            st.markdown("---")
            st.markdown("### 🎯 Decisión del Estudiante")
            
            modelo_elegido = st.radio(
                "Basándote en las métricas, ¿qué modelo elegirías para este pronóstico?",
                options=['Auto-ARIMA', 'ETS'],
                key="radio_modelo_elegido"
            )
            
            if st.button("✅ Confirmar Elección", key="btn_confirmar_modelo"):
                # Calcular votos
                votos = [mejor_rmse, mejor_aic, mejor_bic]
                modelo_mayoria = max(set(votos), key=votos.count)
                
                if modelo_elegido == modelo_mayoria:
                    st.success(f"🎉 ¡Excelente elección! {modelo_elegido} tiene mejor desempeño en {votos.count(modelo_mayoria)}/3 métricas.")
                    st.session_state.puntos_macro += 15
                    st.balloons()
                else:
                    st.warning(f"🤔 {modelo_mayoria} tiene mejor desempeño en {votos.count(modelo_mayoria)}/3 métricas, pero tu elección también es válida según el contexto.")
                    st.session_state.puntos_macro += 10
                
                st.markdown(f"""
                **Análisis de tu elección:**
                - Elegiste: **{modelo_elegido}**
                - RMSE de {modelo_elegido}: {arima_res['rmse'] if modelo_elegido == 'Auto-ARIMA' else ets_res['rmse']:.4f}
                - AIC de {modelo_elegido}: {arima_res['aic'] if modelo_elegido == 'Auto-ARIMA' else ets_res['aic']:.2f}
                - BIC de {modelo_elegido}: {arima_res['bic'] if modelo_elegido == 'Auto-ARIMA' else ets_res['bic']:.2f}
                """)
        
        st.divider()
        st.success(f"🎯 Puntos en Riesgo Macroeconómico: {st.session_state.puntos_macro}")
        
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'Variables Macroeconómicas.xlsx'")
        st.info("💡 Asegúrate de que el archivo esté en el mismo directorio que la aplicación.")
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")


# ============================================================================
# NAVEGACIÓN PRINCIPAL
# ============================================================================

def main():
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs([
        "🌐 Riesgo de Mercado",
        "💼 Riesgo Financiero (Altman)",
        "📈 Riesgo Macroeconómico"
    ])
    
    with tab1:
        tab_riesgo_mercado()
    
    with tab2:
        tab_riesgo_financiero()
    
    with tab3:
        tab_riesgo_macro()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🎓 Laboratorio - Diplomado de mercado de valores y estrategias de inversión</p>
        <p>Desarrollado por Bolsa de Valores Quito para el aprendizaje interactivo</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
