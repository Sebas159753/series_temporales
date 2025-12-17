import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, coint_johansen
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laboratorio de Series de Tiempo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    h2, h3 { color: #34495e; }
    .stButton>button { background-color: #2980b9; color: white; width: 100%; }
    .metric-card { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); text-align: center; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Funciones de Utilidad
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        return df
    except Exception as e:
        return None

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[0], result[1]  # ADF Statistic, p-value

# -----------------------------------------------------------------------------
# Estado de la Sesión
# -----------------------------------------------------------------------------
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_var' not in st.session_state:
    st.session_state.target_var = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'transformed_train' not in st.session_state:
    st.session_state.transformed_train = None
if 'model_fit' not in st.session_state:
    st.session_state.model_fit = None
if 'forecast_res' not in st.session_state:
    st.session_state.forecast_res = None
if 'transform_type' not in st.session_state:
    st.session_state.transform_type = "Ninguna"

# -----------------------------------------------------------------------------
# Sidebar - Navegación Global
# -----------------------------------------------------------------------------
st.sidebar.title("🔬 Laboratorio")

analysis_type = st.sidebar.selectbox(
    "Tipo de Análisis",
    ["Univariante (ARIMA/SARIMA)", "Multivariante (VAR/VECM)"]
)

# Resetear estado si cambia el tipo de análisis
if 'last_analysis_type' not in st.session_state:
    st.session_state.last_analysis_type = analysis_type

if st.session_state.last_analysis_type != analysis_type:
    st.session_state.step = 1
    st.session_state.data = None
    st.session_state.train_data = None
    st.session_state.test_data = None
    st.session_state.model_fit = None
    st.session_state.last_analysis_type = analysis_type
    st.rerun()

# -----------------------------------------------------------------------------
# Lógica: Univariante
# -----------------------------------------------------------------------------
if analysis_type == "Univariante (ARIMA/SARIMA)":
    steps = {
        1: "1. Análisis Exploratorio",
        2: "2. Dividir la serie de tiempo",
        3: "3. Prueba de estacionariedad",
        4: "4. Trasformar la serie (de ser necesario)",
        5: "5. Construir el modelo",
        6: "6. Diagnosis del modelo",
        7: "7. Realizar pronósticos con el modelo",
        8: "8. Transformación inversa del pronóstico",
        9: "9. Realizar la evaluación del pronóstico",
        10: "10. Pronóstico Futuro"
    }

    selected_step = st.sidebar.radio("Pasos del Laboratorio:", list(steps.values()), index=st.session_state.step - 1)
    current_step_num = int(selected_step.split(".")[0])
    st.session_state.step = current_step_num

    st.title(f"🧬 {steps[current_step_num]}")

    # Carga de datos global (siempre visible o accesible)
    file_name = "Variables Macroeconómicas.xlsx"
    if not os.path.exists(file_name):
        st.error(f"No se encontró '{file_name}'.")
        st.stop()

    xl = pd.ExcelFile(file_name, engine="openpyxl")
    sheet_names = xl.sheet_names

    # --- PASO 1: Análisis Exploratorio ---
    if current_step_num == 1:
        st.info("Objetivo: Cargar los datos, visualizar la serie temporal y entender su estructura (tendencia, estacionalidad).")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_sheet = st.selectbox("Selecciona Hoja:", sheet_names)
            df_raw = load_dataset(file_name, selected_sheet)
            
            if df_raw is not None:
                # Detectar fecha
                date_cols = [c for c in df_raw.columns if any(x in str(c).lower() for x in ['fecha', 'date', 'time'])]
                date_col = date_cols[0] if date_cols else st.selectbox("Columna Fecha:", df_raw.columns)
                
                df_raw[date_col] = pd.to_datetime(df_raw[date_col])
                df_raw = df_raw.sort_values(by=date_col).set_index(date_col)
                
                numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
                target_var = st.selectbox("Variable a Analizar:", numeric_cols)
                st.session_state.target_var = target_var
                st.session_state.data = df_raw[[target_var]].dropna()
                
        with col2:
            if st.session_state.data is not None:
                st.subheader("Gráfico de la Serie")
                st.line_chart(st.session_state.data)
                
                st.subheader("Correlograma (ACF y PACF)")
                lags = st.slider("Lags:", 10, 50, 20)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                sm.graphics.tsa.plot_acf(st.session_state.data, lags=lags, ax=ax1)
                sm.graphics.tsa.plot_pacf(st.session_state.data, lags=lags, ax=ax2)
                st.pyplot(fig)

    # --- PASO 2: Dividir la Serie ---
    elif current_step_num == 2:
        if st.session_state.data is None:
            st.warning("Por favor completa el Paso 1 primero.")
        else:
            st.info("Objetivo: Separar los datos en Entrenamiento (para crear el modelo) y Prueba (para evaluar).")
            
            split_pct = st.slider("Porcentaje de Entrenamiento:", 0.5, 0.95, 0.8)
            split_idx = int(len(st.session_state.data) * split_pct)
            
            train = st.session_state.data.iloc[:split_idx]
            test = st.session_state.data.iloc[split_idx:]
            
            st.session_state.train_data = train
            st.session_state.test_data = test
            
            st.write(f"**Train:** {len(train)} observaciones | **Test:** {len(test)} observaciones")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(train, label='Entrenamiento')
            ax.plot(test, label='Prueba', color='orange')
            ax.legend()
            st.pyplot(fig)

    # --- PASO 3: Prueba de Estacionariedad ---
    elif current_step_num == 3:
        if st.session_state.train_data is None:
            st.warning("Por favor completa el Paso 2 primero.")
        else:
            st.info("Objetivo: Verificar si la serie de entrenamiento es estacionaria (media y varianza constantes).")
            
            st.subheader("Test de Dickey-Fuller Aumentado (ADF)")
            if st.button("Ejecutar Test ADF"):
                adf_stat, p_value = check_stationarity(st.session_state.train_data)
                
                col1, col2 = st.columns(2)
                col1.metric("Estadístico ADF", f"{adf_stat:.4f}")
                col2.metric("P-valor", f"{p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ P-valor < 0.05. Rechazamos H0. La serie ES Estacionaria.")
                else:
                    st.error("❌ P-valor >= 0.05. No rechazamos H0. La serie NO es Estacionaria (tiene raíz unitaria).")
                    st.markdown("**Sugerencia:** Ve al siguiente paso para transformar la serie.")

    # --- PASO 4: Transformar la Serie ---
    elif current_step_num == 4:
        if st.session_state.train_data is None:
            st.warning("Por favor completa el Paso 2 primero.")
        else:
            st.info("Objetivo: Aplicar transformaciones (Diferenciación, Logaritmo) para volver estacionaria la serie.")
            
            trans_option = st.radio("Transformación:", ["Ninguna", "Primera Diferencia", "Logaritmo", "Log + Diferencia", "Segunda Diferencia", "Log + Segunda Diferencia"])
            st.session_state.transform_type = trans_option
            
            df_trans = st.session_state.train_data.copy()
            
            if trans_option == "Primera Diferencia":
                df_trans = df_trans.diff().dropna()
            elif trans_option == "Logaritmo":
                df_trans = np.log(df_trans).dropna()
            elif trans_option == "Log + Diferencia":
                df_trans = np.log(df_trans).diff().dropna()
            elif trans_option == "Segunda Diferencia":
                df_trans = df_trans.diff().diff().dropna()
            elif trans_option == "Log + Segunda Diferencia":
                df_trans = np.log(df_trans).diff().diff().dropna()
                
            st.session_state.transformed_train = df_trans
            
            st.line_chart(df_trans)
            
            st.markdown("### Verificación Post-Transformación")
            adf_stat, p_value = check_stationarity(df_trans)
            st.write(f"**Nuevo P-valor:** {p_value:.4f}")
            if p_value < 0.05:
                st.success("La serie transformada es Estacionaria.")
            else:
                st.warning("La serie aún podría no ser estacionaria. Prueba otra transformación.")

    # --- PASO 5: Construir el Modelo ---
    elif current_step_num == 5:
        if st.session_state.transformed_train is None:
            st.warning("Por favor completa el Paso 4 primero.")
        else:
            st.info("Objetivo: Definir los parámetros p, d, q y ajustar el modelo ARIMA.")
            
            # Mostrar modelo seleccionado actualmente si existe
            if st.session_state.model_fit is not None:
                try:
                    # Intentar obtener el orden del modelo ajustado
                    current_order = st.session_state.model_fit.model.order
                    current_seasonal = st.session_state.model_fit.model.seasonal_order
                    
                    if current_seasonal and current_seasonal != (0,0,0,0):
                        st.success(f"✅ Modelo Actual Seleccionado: SARIMA{current_order}x{current_seasonal}")
                    else:
                        st.success(f"✅ Modelo Actual Seleccionado: ARIMA{current_order}")
                except:
                    st.success("✅ Modelo Actual Seleccionado")
            
            # Lógica para ocultar/fijar 'd' si ya se diferenció
            already_diff = st.session_state.transform_type in ["Primera Diferencia", "Log + Diferencia", "Segunda Diferencia", "Log + Segunda Diferencia"]
            
            st.markdown("### 1. Configuración de Estacionalidad")
            st.write("Antes de buscar el modelo, definamos si la serie tiene un componente estacional.")
            
            col_s1, col_s2 = st.columns([1, 2])
            
            with col_s1:
                use_seasonal = st.checkbox("¿La serie tiene estacionalidad?", value=True)
                
            seasonal_period = 12
            if use_seasonal:
                with col_s2:
                    # Auto-detección de periodo
                    if st.button("🔍 Detectar Periodo Automáticamente"):
                        # Calcular ACF
                        acf_vals = acf(st.session_state.transformed_train, nlags=40)
                        candidates = [4, 12, 24]
                        best_s = 12
                        max_corr = 0
                        for cand in candidates:
                            if cand < len(acf_vals):
                                if abs(acf_vals[cand]) > max_corr:
                                    max_corr = abs(acf_vals[cand])
                                    best_s = cand
                        st.session_state.suggested_period = best_s
                        st.success(f"Periodo detectado: {best_s}")
                    
                    default_s = st.session_state.get('suggested_period', 12)
                    seasonal_period = st.number_input("Periodo Estacional (s):", 2, 24, default_s)
                    st.caption("Ejemplo: 12 para datos mensuales, 4 para trimestrales.")

            st.markdown("---")
            st.markdown("### 2. Búsqueda del Mejor Modelo")
            st.write("El sistema buscará automáticamente la mejor combinación de parámetros (ARIMA o SARIMA).")
            
            if st.button("🚀 Iniciar Búsqueda Automática"):
                progress_bar = st.progress(0)
                
                # Configuración del Grid
                # p, q: 0 a 3
                # d: 0 a 1 (o fijo en 0 si ya diferenciado)
                # P, Q: 0 a 1 (si estacional)
                # D: 0 a 1 (si estacional)
                
                max_p = 3
                max_q = 3
                
                d_options = [0] if already_diff else [0, 1]
                
                combinations = [(p, d, q) for p in range(max_p + 1) for d in d_options for q in range(max_q + 1)]
                
                seasonal_combs = [(0,0,0,0)]
                if use_seasonal:
                    # Grid estacional ligero: P=[0,1], D=[0,1], Q=[0,1]
                    seasonal_combs = [(P, D, Q, seasonal_period) for P in [0, 1] for D in [0, 1] for Q in [0, 1]]
                
                total_combs = len(combinations) * len(seasonal_combs)
                results_list = []
                counter = 0
                
                status_text = st.empty()
                
                for order in combinations:
                    for seas_order in seasonal_combs:
                        counter += 1
                        status_text.text(f"Probando: ARIMA{order} x {seas_order}...")
                        try:
                            model = ARIMA(st.session_state.transformed_train, order=order, seasonal_order=seas_order)
                            res = model.fit()
                            
                            rmse_in = np.sqrt(mean_squared_error(st.session_state.transformed_train, res.fittedvalues))
                            
                            results_list.append({
                                "Order": str(order),
                                "Seasonal": str(seas_order),
                                "AIC": res.aic,
                                "BIC": res.bic,
                                "RMSE": rmse_in,
                                "p": order[0], "d": order[1], "q": order[2],
                                "P": seas_order[0], "D": seas_order[1], "Q": seas_order[2], "s": seas_order[3]
                            })
                        except:
                            continue
                        
                        if counter % 5 == 0:
                            progress_bar.progress(min(counter / total_combs, 1.0))
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                if results_list:
                    df_res = pd.DataFrame(results_list)
                    st.session_state.grid_results = df_res
                    st.success("✅ Búsqueda completada. Selecciona un modelo abajo.")
                else:
                    st.error("❌ No se encontraron modelos válidos.")
            
            if 'grid_results' in st.session_state:
                df_res = st.session_state.grid_results
                
                # Identificar mejores modelos
                best_aic_row = df_res.loc[df_res['AIC'].idxmin()]
                best_rmse_row = df_res.loc[df_res['RMSE'].idxmin()]
                
                st.subheader("Resultados de la Búsqueda")
                st.dataframe(df_res.sort_values(by="AIC").head(10).style.format({"AIC": "{:.2f}", "BIC": "{:.2f}", "RMSE": "{:.4f}"}))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"🏆 Mejor AIC: ARIMA{best_aic_row['Order']} x {best_aic_row['Seasonal']}")
                    st.metric("AIC", f"{best_aic_row['AIC']:.2f}")
                    st.metric("RMSE", f"{best_aic_row['RMSE']:.4f}")
                    if st.button("Seleccionar Modelo (Min AIC)"):
                        p, d, q = int(best_aic_row['p']), int(best_aic_row['d']), int(best_aic_row['q'])
                        P, D, Q, s = int(best_aic_row['P']), int(best_aic_row['D']), int(best_aic_row['Q']), int(best_aic_row['s'])
                        
                        model = ARIMA(st.session_state.transformed_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
                        st.session_state.model_fit = model.fit()
                        st.success(f"Modelo seleccionado y entrenado.")
                        st.rerun()

                with col2:
                    st.info(f"🎯 Mejor RMSE: ARIMA{best_rmse_row['Order']} x {best_rmse_row['Seasonal']}")
                    st.metric("AIC", f"{best_rmse_row['AIC']:.2f}")
                    st.metric("RMSE", f"{best_rmse_row['RMSE']:.4f}")
                    if st.button("Seleccionar Modelo (Min RMSE)"):
                        p, d, q = int(best_rmse_row['p']), int(best_rmse_row['d']), int(best_rmse_row['q'])
                        P, D, Q, s = int(best_rmse_row['P']), int(best_rmse_row['D']), int(best_rmse_row['Q']), int(best_rmse_row['s'])
                        
                        model = ARIMA(st.session_state.transformed_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
                        st.session_state.model_fit = model.fit()
                        st.success(f"Modelo seleccionado y entrenado.")
                        st.rerun()

    # --- PASO 6: Diagnosis del Modelo ---
    elif current_step_num == 6:
        if st.session_state.model_fit is None:
            st.warning("Por favor entrena el modelo en el Paso 5.")
        else:
            st.info("Objetivo: Analizar los residuos para asegurar que se comportan como Ruido Blanco.")
            
            model_fit = st.session_state.model_fit
            residuals = model_fit.resid
            
            st.subheader("Gráficos de Diagnóstico")
            fig = model_fit.plot_diagnostics(figsize=(10, 8))
            st.pyplot(fig)
            
            st.subheader("Test de Ljung-Box (Autocorrelación de residuos)")
            lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
            p_val_lb = lb_test['lb_pvalue'].values[0]
            st.write(f"**P-valor (lag 10):** {p_val_lb:.4f}")
            
            if p_val_lb > 0.05:
                st.success("✅ P-valor > 0.05. No rechazamos H0. Los residuos son independientes (Buen modelo).")
            else:
                st.error("❌ P-valor < 0.05. Los residuos tienen autocorrelación. Ajusta p o q en el Paso 5.")

    # --- PASO 7: Realizar Pronósticos ---
    elif current_step_num == 7:
        if st.session_state.model_fit is None:
            st.warning("Modelo no entrenado.")
        else:
            st.info("Objetivo: Generar predicciones para el horizonte de prueba.")
            
            steps_forecast = len(st.session_state.test_data)
            forecast_res = st.session_state.model_fit.get_forecast(steps=steps_forecast)
            st.session_state.forecast_res = forecast_res
            
            pred_mean = forecast_res.predicted_mean
            pred_ci = forecast_res.conf_int()
            
            # Ajustar índice de predicción al de test para graficar
            pred_mean.index = st.session_state.test_data.index
            pred_ci.index = st.session_state.test_data.index
            
            st.subheader("Pronóstico (Escala Transformada)")
            fig, ax = plt.subplots(figsize=(10, 4))
            # Mostrar ultimos datos de train transformados para contexto
            ax.plot(st.session_state.transformed_train.iloc[-50:], label='Historia (Transf.)')
            ax.plot(pred_mean, label='Pronóstico', color='red')
            ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
            ax.legend()
            st.pyplot(fig)

    # --- PASO 8: Transformación Inversa ---
    elif current_step_num == 8:
        if st.session_state.forecast_res is None:
            st.warning("Genera el pronóstico en el Paso 7.")
        else:
            st.info("Objetivo: Revertir las transformaciones para comparar con los datos reales.")
            
            pred_mean = st.session_state.forecast_res.predicted_mean
            pred_mean.index = st.session_state.test_data.index
            
            # Lógica de inversión
            final_forecast = None
            
            if st.session_state.transform_type == "Ninguna":
                final_forecast = pred_mean
                
            elif st.session_state.transform_type == "Logaritmo":
                final_forecast = np.exp(pred_mean)
                
            elif st.session_state.transform_type == "Primera Diferencia":
                # Reconstruir desde el último valor de train
                last_val = st.session_state.train_data.iloc[-1].values[0]
                final_forecast = last_val + pred_mean.cumsum()
                
            elif st.session_state.transform_type == "Log + Diferencia":
                # Reconstruir log-diff -> log -> exp
                last_log_val = np.log(st.session_state.train_data.iloc[-1].values[0])
                log_forecast = last_log_val + pred_mean.cumsum()
                final_forecast = np.exp(log_forecast)

            elif st.session_state.transform_type == "Segunda Diferencia":
                # Reconstruir 2da diff -> 1ra diff -> niveles
                last_val = st.session_state.train_data.iloc[-1].values[0]
                last_diff = st.session_state.train_data.diff().iloc[-1].values[0]
                
                # 1. Reconstruir 1ra diferencia
                forecast_diff1 = last_diff + pred_mean.cumsum()
                # 2. Reconstruir niveles
                final_forecast = last_val + forecast_diff1.cumsum()
                
            elif st.session_state.transform_type == "Log + Segunda Diferencia":
                # Reconstruir log-2da-diff -> log-1ra-diff -> log-niveles -> exp
                log_train = np.log(st.session_state.train_data)
                last_log_val = log_train.iloc[-1].values[0]
                last_log_diff = log_train.diff().iloc[-1].values[0]
                
                # 1. Reconstruir 1ra diferencia de logs
                forecast_log_diff1 = last_log_diff + pred_mean.cumsum()
                # 2. Reconstruir niveles de logs
                forecast_log_levels = last_log_val + forecast_log_diff1.cumsum()
                # 3. Exponencial
                final_forecast = np.exp(forecast_log_levels)
                
            st.session_state.final_forecast = final_forecast
            
            st.write(f"**Transformación aplicada:** {st.session_state.transform_type}")
            st.success("Transformación inversa calculada.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Pronóstico Transformado (Head):")
                st.write(pred_mean.head())
            with col2:
                st.write("Pronóstico Original (Head):")
                st.write(final_forecast.head())

    # --- PASO 9: Evaluación ---
    elif current_step_num == 9:
        if 'final_forecast' not in st.session_state:
            st.warning("Realiza la transformación inversa en el Paso 8.")
        else:
            st.info("Objetivo: Comparar el pronóstico final con los datos reales reservados (Test).")
            
            test_vals = st.session_state.test_data.iloc[:, 0]
            pred_vals = st.session_state.final_forecast
            
            # Métricas
            rmse = np.sqrt(mean_squared_error(test_vals, pred_vals))
            mae = mean_absolute_error(test_vals, pred_vals)
            mape = np.mean(np.abs((test_vals - pred_vals) / test_vals)) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("MAE", f"{mae:.4f}")
            col3.metric("MAPE", f"{mape:.2f}%")
            
            st.subheader("Gráfico Final: Real vs Pronóstico")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(st.session_state.train_data.iloc[-100:], label='Entrenamiento')
            ax.plot(st.session_state.test_data, label='Datos Reales (Test)', color='green')
            ax.plot(pred_vals, label='Pronóstico Modelo', color='red', linestyle='--')
            ax.set_title(f"Pronóstico con {st.session_state.transform_type}")
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("Tabla de Resultados: Real vs Pronóstico")
            results_df = pd.DataFrame({
                "Real": test_vals,
                "Pronóstico": pred_vals,
                "Error": test_vals - pred_vals,
                "Error Absoluto": np.abs(test_vals - pred_vals)
            })
            st.dataframe(results_df)
            
            st.success("¡Evaluación Completada!")

    # --- PASO 10: Pronóstico Futuro ---
    elif current_step_num == 10:
        if st.session_state.model_fit is None:
            st.warning("Debes construir y seleccionar un modelo en el Paso 5 primero.")
        else:
            st.info("Objetivo: Utilizar toda la información disponible (Train + Test) para predecir el futuro.")
            
            st.subheader("Configuración del Pronóstico Futuro")
            future_steps = st.slider("Pasos a futuro:", 1, 24, 12)
            
            if st.button("Generar Pronóstico Futuro"):
                # 1. Re-entrenar con TODOS los datos
                full_data = st.session_state.data.iloc[:, 0]
                
                # Aplicar transformación a toda la serie
                full_data_trans = full_data.copy()
                if st.session_state.transform_type == "Primera Diferencia":
                    full_data_trans = full_data_trans.diff().dropna()
                elif st.session_state.transform_type == "Logaritmo":
                    full_data_trans = np.log(full_data_trans).dropna()
                elif st.session_state.transform_type == "Log + Diferencia":
                    full_data_trans = np.log(full_data_trans).diff().dropna()
                elif st.session_state.transform_type == "Segunda Diferencia":
                    full_data_trans = full_data_trans.diff().diff().dropna()
                elif st.session_state.transform_type == "Log + Segunda Diferencia":
                    full_data_trans = np.log(full_data_trans).diff().diff().dropna()
                
                # Obtener orden del modelo actual
                try:
                    order = st.session_state.model_fit.model.order
                    seasonal_order = st.session_state.model_fit.model.seasonal_order
                except:
                    st.error("No se pudo recuperar el orden del modelo.")
                    st.stop()
                    
                # Ajustar modelo final
                model_future = ARIMA(full_data_trans, order=order, seasonal_order=seasonal_order)
                res_future = model_future.fit()
                
                # Pronosticar
                forecast_res = res_future.get_forecast(steps=future_steps)
                pred_mean = forecast_res.predicted_mean
                pred_ci = forecast_res.conf_int()
                
                # Inversión de transformación para el futuro
                final_future_forecast = None
                
                if st.session_state.transform_type == "Ninguna":
                    final_future_forecast = pred_mean
                    
                elif st.session_state.transform_type == "Logaritmo":
                    final_future_forecast = np.exp(pred_mean)
                    
                elif st.session_state.transform_type == "Primera Diferencia":
                    last_val = full_data.iloc[-1]
                    final_future_forecast = last_val + pred_mean.cumsum()
                    
                elif st.session_state.transform_type == "Log + Diferencia":
                    last_log_val = np.log(full_data.iloc[-1])
                    log_forecast = last_log_val + pred_mean.cumsum()
                    final_future_forecast = np.exp(log_forecast)

                elif st.session_state.transform_type == "Segunda Diferencia":
                    last_val = full_data.iloc[-1]
                    last_diff = full_data.diff().iloc[-1]
                    forecast_diff1 = last_diff + pred_mean.cumsum()
                    final_future_forecast = last_val + forecast_diff1.cumsum()
                    
                elif st.session_state.transform_type == "Log + Segunda Diferencia":
                    log_full = np.log(full_data)
                    last_log_val = log_full.iloc[-1]
                    last_log_diff = log_full.diff().iloc[-1]
                    forecast_log_diff1 = last_log_diff + pred_mean.cumsum()
                    forecast_log_levels = last_log_val + forecast_log_diff1.cumsum()
                    final_future_forecast = np.exp(forecast_log_levels)
                
                # Gráfico
                st.subheader("Proyección a Futuro")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(full_data.iloc[-100:], label='Historia Reciente')
                ax.plot(final_future_forecast, label='Pronóstico Futuro', color='purple', linestyle='--')
                ax.set_title(f"Pronóstico Futuro (Modelo ARIMA{order} x {seasonal_order})")
                ax.legend()
                st.pyplot(fig)
                
                st.write("Valores Pronosticados:")
                st.dataframe(final_future_forecast)

# -----------------------------------------------------------------------------
# Lógica: Multivariante (VAR/VECM)
# -----------------------------------------------------------------------------
elif analysis_type == "Multivariante (VAR/VECM)":
    steps = {
        1: "1. Análisis Exploratorio (Multivariante)",
        2: "2. Dividir Series (Train/Test)",
        3: "3. Prueba de Estacionariedad (Múltiple)",
        4: "4. Cointegración y Transformación",
        5: "5. Construir Modelo (VAR/VECM)",
        6: "6. Diagnosis del Modelo",
        7: "7. Impulso-Respuesta (IRF)",
        8: "8. Pronóstico y Evaluación",
        9: "9. Pronóstico Futuro"
    }
    
    selected_step = st.sidebar.radio("Pasos del Laboratorio:", list(steps.values()), index=st.session_state.step - 1)
    current_step_num = int(selected_step.split(".")[0])
    st.session_state.step = current_step_num

    st.title(f"🧬 {steps[current_step_num]}")
    
    file_name = "Variables Macroeconómicas.xlsx"
    if not os.path.exists(file_name):
        st.error(f"No se encontró '{file_name}'.")
        st.stop()

    xl = pd.ExcelFile(file_name, engine="openpyxl")
    sheet_names = xl.sheet_names
    
    # --- PASO 1: Análisis Exploratorio (Multi) ---
    if current_step_num == 1:
        st.info("Objetivo: Seleccionar múltiples variables y analizar sus relaciones (correlación, tendencias).")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_sheet = st.selectbox("Selecciona Hoja:", sheet_names)
            df_raw = load_dataset(file_name, selected_sheet)
            
            if df_raw is not None:
                date_cols = [c for c in df_raw.columns if any(x in str(c).lower() for x in ['fecha', 'date', 'time'])]
                date_col = date_cols[0] if date_cols else st.selectbox("Columna Fecha:", df_raw.columns)
                
                df_raw[date_col] = pd.to_datetime(df_raw[date_col])
                df_raw = df_raw.sort_values(by=date_col).set_index(date_col)
                
                numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
                
                target_vars = st.multiselect("Variables a Analizar (Min 2):", numeric_cols, default=numeric_cols[:2] if len(numeric_cols)>=2 else numeric_cols)
                
                if len(target_vars) < 2:
                    st.warning("Selecciona al menos 2 variables para análisis multivariante.")
                else:
                    st.session_state.target_vars = target_vars
                    st.session_state.data = df_raw[target_vars].dropna()
        
        with col2:
            if st.session_state.data is not None and len(st.session_state.data.columns) >= 2:
                st.subheader("Gráfico de Series")
                st.line_chart(st.session_state.data)
                
                st.subheader("Matriz de Correlación")
                corr = st.session_state.data.corr()
                st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
                
    # --- PASO 2: Dividir Series ---
    elif current_step_num == 2:
        if st.session_state.data is None:
            st.warning("Completa el Paso 1.")
        else:
            st.info("Objetivo: Separar el conjunto de datos multivariante en Train y Test.")
            
            split_pct = st.slider("Porcentaje de Entrenamiento:", 0.5, 0.95, 0.8)
            split_idx = int(len(st.session_state.data) * split_pct)
            
            train = st.session_state.data.iloc[:split_idx]
            test = st.session_state.data.iloc[split_idx:]
            
            st.session_state.train_data = train
            st.session_state.test_data = test
            
            st.write(f"**Train:** {len(train)} | **Test:** {len(test)}")
            
            st.line_chart(train)
            st.caption("Datos de Entrenamiento")

    # --- PASO 3: Estacionariedad (Multi) ---
    elif current_step_num == 3:
        if st.session_state.train_data is None:
            st.warning("Completa el Paso 2.")
        else:
            st.info("Objetivo: Verificar estacionariedad de CADA variable individualmente.")
            
            if st.button("Ejecutar Test ADF para todas las variables"):
                results = []
                for col in st.session_state.train_data.columns:
                    stat, p_val = check_stationarity(st.session_state.train_data[col])
                    is_stat = p_val < 0.05
                    results.append({
                        "Variable": col,
                        "ADF Stat": stat,
                        "P-valor": p_val,
                        "¿Estacionaria?": "✅ Sí" if is_stat else "❌ No"
                    })
                
                st.table(pd.DataFrame(results))
                
                if any(r["P-valor"] >= 0.05 for r in results):
                    st.warning("Algunas variables NO son estacionarias. Deberás diferenciarlas o usar VECM si cointegran.")
                else:
                    st.success("Todas las variables son estacionarias. Podrías usar VAR directamente.")

    # --- PASO 4: Cointegración y Transformación ---
    elif current_step_num == 4:
        if st.session_state.train_data is None:
            st.warning("Completa el Paso 2.")
        else:
            st.info("Objetivo: Determinar si las series cointegran (relación a largo plazo) y aplicar transformaciones si es necesario.")
            


            st.subheader("1. Transformación de Datos")
            st.write("Selecciona una transformación para aplicar a las series antes de realizar el test de cointegración.")
            
            trans_option = st.radio("Transformación:", ["Ninguna", "Primera Diferencia", "Logaritmo", "Log + Diferencia", "Segunda Diferencia"], key="multi_trans")
            st.session_state.transform_type = trans_option
            
            df_trans = st.session_state.train_data.copy()
            
            if trans_option == "Primera Diferencia":
                df_trans = df_trans.diff().dropna()
            elif trans_option == "Logaritmo":
                df_trans = np.log(df_trans).dropna()
            elif trans_option == "Log + Diferencia":
                df_trans = np.log(df_trans).diff().dropna()
            elif trans_option == "Segunda Diferencia":
                df_trans = df_trans.diff().diff().dropna()
                
            st.session_state.transformed_train = df_trans
            st.line_chart(df_trans)
            
            st.markdown("---")
            st.subheader("2. Test de Cointegración de Johansen")
            st.write(f"Este test se ejecutará sobre los datos transformados ({trans_option}).")
            
            if st.button("Ejecutar Test de Johansen"):
                try:
                    # Johansen test on TRANSFORMED data
                    # det_order=0 (constant term), k_ar_diff=1 (lags)
                    joh_res = coint_johansen(df_trans, det_order=0, k_ar_diff=1)
                    
                    # Display Trace Statistic
                    st.write("**Estadístico de la Traza (Trace Statistic)**")
                    trace_df = pd.DataFrame(joh_res.lr1, index=[f"r <= {i}" for i in range(len(joh_res.lr1))], columns=["Trace Stat"])
                    trace_df["Critical Val (90%)"] = joh_res.cvt[:, 0]
                    trace_df["Critical Val (95%)"] = joh_res.cvt[:, 1]
                    trace_df["Critical Val (99%)"] = joh_res.cvt[:, 2]
                    trace_df["¿Cointegra?"] = trace_df["Trace Stat"] > trace_df["Critical Val (95%)"]
                    st.dataframe(trace_df)
                    
                    # Interpretation
                    r_count = trace_df["¿Cointegra?"].sum()
                    if r_count > 0:
                        st.success(f"✅ Se encontraron {r_count} relaciones de cointegración (al 95%). Sugerencia: Usar modelo VECM.")
                    else:
                        st.warning("❌ No se encontró cointegración. Sugerencia: Diferenciar las series y usar VAR.")
                        
                except Exception as e:
                    st.error(f"Error al ejecutar Johansen: {e}")
            


    # --- PASO 5: Selección y Estimación del Modelo (VAR / VECM) ---
    elif current_step_num == 5:
        if 'transformed_train' not in st.session_state or st.session_state.transformed_train is None:
            st.warning("Completa el Paso 4 (Transformación).")
        else:
            st.info("Objetivo: Seleccionar el tipo de modelo (VAR o VECM), determinar el orden de rezagos (lags) y estimar el modelo.")
            
            model_type = st.radio("Selecciona el Tipo de Modelo:", ["VAR", "VECM"], key="multi_model_type")
            
            data_for_model = st.session_state.transformed_train
            
            if model_type == "VAR":
                st.subheader("Modelo VAR (Vectores Autorregresivos)")
                st.write("Adecuado para series estacionarias sin cointegración.")
                
                # Lag Order Selection
                max_lags = st.slider("Máximo de Rezagos a evaluar:", 1, 20, 10)
                if st.button("Seleccionar Orden de Rezagos (VAR)"):
                    try:
                        model_var = VAR(data_for_model)
                        lag_order_results = model_var.select_order(maxlags=max_lags)
                        st.write(lag_order_results.summary())
                        st.session_state.suggested_lags = lag_order_results.aic
                        st.success(f"Rezago sugerido (AIC): {st.session_state.suggested_lags}")
                    except Exception as e:
                        st.error(f"Error al seleccionar rezagos: {e}")
                
                lags = st.number_input("Ingresa el número de rezagos (p) para el modelo:", min_value=1, value=st.session_state.get('suggested_lags', 1))
                
                if st.button("Entrenar Modelo VAR"):
                    try:
                        model_var = VAR(data_for_model)
                        results_var = model_var.fit(lags)
                        st.session_state.model_fit = results_var
                        st.session_state.model_type = "VAR"
                        st.session_state.lags = lags
                        st.success(f"Modelo VAR({lags}) entrenado exitosamente.")
                        st.write(results_var.summary())
                    except Exception as e:
                        st.error(f"Error al entrenar VAR: {e}")

            elif model_type == "VECM":
                st.subheader("Modelo VECM (Vector Error Correction Model)")
                st.write("Adecuado para series cointegradas.")
                
                # VECM Parameters
                # k_ar_diff = lags - 1. So if VAR lag is p, VECM lag is p-1.
                # We usually select lag order on the underlying VAR representation.
                
                st.write("Selección de Rezagos (basado en VAR subyacente):")
                max_lags_vecm = st.slider("Máximo de Rezagos a evaluar (VAR):", 1, 20, 10, key="vecm_lags")
                if st.button("Seleccionar Orden de Rezagos (VECM)"):
                     try:
                        model_var = VAR(data_for_model)
                        lag_order_results = model_var.select_order(maxlags=max_lags_vecm)
                        st.write(lag_order_results.summary())
                        st.session_state.suggested_lags_vecm = lag_order_results.aic
                        st.success(f"Rezago sugerido (AIC) para VAR: {st.session_state.suggested_lags_vecm}. Para VECM usar k_ar_diff = {st.session_state.suggested_lags_vecm - 1}")
                     except Exception as e:
                        st.error(f"Error al seleccionar rezagos: {e}")

                k_ar_diff = st.number_input("Rezagos de diferencias (k_ar_diff):", min_value=0, value=st.session_state.get('suggested_lags_vecm', 2) - 1)
                coint_rank = st.number_input("Rango de Cointegración (r):", min_value=1, value=1)
                det_order_vecm = st.selectbox("Orden Determinista:", ["n", "co", "ci", "lo", "li"], index=1, help="n: no constant, co: constant outside, ci: constant inside, etc.")

                if st.button("Entrenar Modelo VECM"):
                    try:
                        # VECM requires un-differenced data usually, but if we transformed log, we use log data.
                        # If we differenced in Step 4, we should probably use the data BEFORE differencing for VECM?
                        # VECM is for non-stationary cointegrated series.
                        # If user selected "Primera Diferencia" in Step 4, they made it stationary, so VECM might not be appropriate on THAT data.
                        # However, for simplicity, we assume user knows what they are doing or we guide them.
                        # Ideally, for VECM, we use the level data (or log data), NOT differenced data.
                        
                        # Let's check if data was differenced in Step 4.
                        if "Diferencia" in st.session_state.transform_type:
                            st.warning("⚠️ Estás usando datos diferenciados para VECM. VECM se aplica usualmente a las series en niveles (o log) que cointegran. Considera cambiar la transformación en el Paso 4 a 'Ninguna' o 'Logaritmo'.")
                        
                        model_vecm = VECM(data_for_model, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=det_order_vecm)
                        results_vecm = model_vecm.fit()
                        st.session_state.model_fit = results_vecm
                        st.session_state.model_type = "VECM"
                        st.session_state.lags = k_ar_diff # Store for reference
                        st.success(f"Modelo VECM(k_ar_diff={k_ar_diff}, r={coint_rank}) entrenado exitosamente.")
                        st.write(results_vecm.summary())
                    except Exception as e:
                        st.error(f"Error al entrenar VECM: {e}")

    # --- PASO 6: Diagnóstico del Modelo ---
    elif current_step_num == 6:
        if 'model_fit' not in st.session_state or st.session_state.model_fit is None:
            st.warning("Completa el Paso 5 (Estimación del Modelo).")
        else:
            st.info("Objetivo: Verificar si los residuos del modelo se comportan como ruido blanco (sin autocorrelación).")
            
            results = st.session_state.model_fit
            model_type = st.session_state.model_type
            
            st.subheader("Análisis de Residuos")
            
            # Residuals DataFrame
            resid = results.resid
            st.write("Primeras filas de los residuos:")
            st.write(resid.head())
            
            if st.button("Ejecutar Diagnóstico"):
                 # 1. Portmanteau Test (Autocorrelation)
                 st.subheader("1. Test de Portmanteau (Autocorrelación)")
                 try:
                     # test_whiteness returns a summary object
                     whiteness = results.test_whiteness(nlags=10, adjust=True)
                     st.text(whiteness.summary())
                     st.info("Interpretación: H0 = No hay autocorrelación (Ruido Blanco). Si Prob > 0.05, ✅ Pasa (Los residuos son ruido blanco).")
                 except Exception as e:
                     st.warning(f"No se pudo ejecutar Portmanteau: {e}")

                 # 2. Normality Test (Jarque-Bera)
                 st.subheader("2. Test de Normalidad (Jarque-Bera)")
                 try:
                     normality = results.test_normality()
                     st.text(normality.summary())
                     st.info("Interpretación: H0 = Los residuos son Normales. Si Prob > 0.05, ✅ Pasa (Distribución Normal).")
                 except Exception as e:
                     st.warning(f"No se pudo ejecutar Jarque-Bera: {e}")
                 
                 # 3. Residual Plots
                 st.subheader("3. Gráficos de Residuos")
                 
                 var_to_plot = st.selectbox("Ver residuos de:", resid.columns)
                 
                 fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                 ax[0].plot(resid[var_to_plot])
                 ax[0].set_title(f"Residuos: {var_to_plot}")
                 
                 plot_acf(resid[var_to_plot], ax=ax[1], lags=20)
                 ax[1].set_title("Autocorrelación (ACF)")
                 st.pyplot(fig)

    # --- PASO 7: Función de Impulso-Respuesta (IRF) ---
    elif current_step_num == 7:
        if 'model_fit' not in st.session_state or st.session_state.model_fit is None:
            st.warning("Completa el Paso 5 (Estimación del Modelo).")
        else:
            st.info("Objetivo: Analizar cómo reacciona una variable ante un choque (impulso) en otra variable a lo largo del tiempo.")
            
            results = st.session_state.model_fit
            model_type = st.session_state.model_type
            
            periods = st.slider("Periodos hacia adelante:", 5, 40, 10)
            
            if st.button("Generar Funciones de Impulso-Respuesta"):
                try:
                    irf = results.irf(periods)
                    st.subheader("Gráficos de Impulso-Respuesta")
                    st.write("Las líneas azules representan la respuesta de la variable ante un choque. Las líneas punteadas son los intervalos de confianza.")
                    
                    # Plotting all or specific
                    # irf.plot() plots all combinations which can be messy if many variables.
                    # Let's allow user to choose or plot all in a big figure.
                    
                    fig = irf.plot(orth=True) # Orthogonalized IRF is standard
                    st.pyplot(fig)
                    
                    # Option to plot specific pair?
                    # irf.plot_cum_effects() also available
                    
                except Exception as e:
                    st.error(f"Error al generar IRF: {e}")

    # --- PASO 8: Pronóstico y Evaluación ---
    elif current_step_num == 8:
        if 'model_fit' not in st.session_state or st.session_state.model_fit is None:
            st.warning("Completa el Paso 5 (Estimación del Modelo).")
        elif st.session_state.test_data is None:
            st.warning("No hay datos de prueba. Revisa el Paso 2.")
        else:
            st.info("Objetivo: Generar pronósticos para el conjunto de prueba y evaluar la precisión del modelo.")
            
            results = st.session_state.model_fit
            model_type = st.session_state.model_type
            n_forecast = len(st.session_state.test_data)
            
            if st.button("Generar Pronóstico", key="btn_forecast_multi"):
                try:
                    with st.spinner("Calculando pronóstico..."):
                        forecast_values = None
                        
                        if model_type == "VAR":
                            # VAR forecasting
                            # We need the last 'lags' observations from the training data (transformed)
                            lag_order = st.session_state.lags
                            forecast_input = st.session_state.transformed_train.values[-lag_order:]
                            fc = results.forecast(y=forecast_input, steps=n_forecast)
                            forecast_df = pd.DataFrame(fc, index=st.session_state.test_data.index, columns=st.session_state.test_data.columns)
                            forecast_values = forecast_df
                            
                        elif model_type == "VECM":
                            # VECM forecasting
                            # predict() method usually takes steps
                            fc = results.predict(steps=n_forecast)
                            forecast_df = pd.DataFrame(fc, index=st.session_state.test_data.index, columns=st.session_state.test_data.columns)
                            forecast_values = forecast_df
                        
                        # Inverse Transformation
                        # We need to invert based on st.session_state.transform_type
                        # This is tricky for multivariate if we did differencing.
                        # If "Primera Diferencia", we need to cumulatively sum starting from the last training point.
                        
                        trans_type = st.session_state.transform_type
                        final_forecast = forecast_values.copy()
                        
                        # Logic for inverse transformation (simplified for now, might need robust implementation like in Univariate)
                        
                        if trans_type == "Primera Diferencia":
                            # We need to cumsum and add the last value of train_data (original scale)
                            # For each column
                            for col in final_forecast.columns:
                                last_val = st.session_state.train_data[col].iloc[-1]
                                final_forecast[col] = final_forecast[col].cumsum() + last_val
                                
                        elif trans_type == "Logaritmo":
                            final_forecast = np.exp(final_forecast)
                            
                        elif trans_type == "Log + Diferencia":
                            # First inverse diff (cumsum + last log val)
                            # Then exp
                             for col in final_forecast.columns:
                                last_log_val = np.log(st.session_state.train_data[col].iloc[-1])
                                final_forecast[col] = final_forecast[col].cumsum() + last_log_val
                             final_forecast = np.exp(final_forecast)
                             
                        elif trans_type == "Segunda Diferencia":
                             # Harder, need last two values. 
                             # For simplicity, let's assume user knows or we implement basic 2nd diff inverse if needed.
                             # But let's stick to 1st diff support mainly.
                             st.warning("La inversión de 'Segunda Diferencia' puede ser compleja y no está totalmente automatizada aquí. Se muestran valores pronosticados tal cual.")

                        st.session_state.forecast_results = final_forecast
                        st.success("Pronóstico generado correctamente.")
                    
                except Exception as e:
                    st.error(f"Error en Pronóstico: {e}")

            if 'forecast_results' in st.session_state and st.session_state.forecast_results is not None:
                final_forecast = st.session_state.forecast_results
                
                st.subheader("Resultados del Pronóstico")
                st.write(final_forecast.head())
                
                # Evaluation Metrics
                st.subheader("Métricas de Evaluación")
                
                metrics_list = []
                for col in st.session_state.test_data.columns:
                    actual = st.session_state.test_data[col]
                    pred = final_forecast[col]
                    
                    rmse = np.sqrt(mean_squared_error(actual, pred))
                    mae = mean_absolute_error(actual, pred)
                    mape = np.mean(np.abs((actual - pred) / actual)) * 100
                    
                    metrics_list.append({"Variable": col, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape})
                    
                metrics_df = pd.DataFrame(metrics_list)
                st.table(metrics_df)
                
                # Plotting
                st.subheader("Gráfico Comparativo (Real vs Pronóstico)")
                var_to_plot_eval = st.selectbox("Selecciona Variable para visualizar:", st.session_state.test_data.columns, key="eval_plot_var")
                
                fig_eval, ax_eval = plt.subplots(figsize=(10, 5))
                ax_eval.plot(st.session_state.train_data.index, st.session_state.train_data[var_to_plot_eval], label="Entrenamiento")
                ax_eval.plot(st.session_state.test_data.index, st.session_state.test_data[var_to_plot_eval], label="Prueba (Real)")
                ax_eval.plot(final_forecast.index, final_forecast[var_to_plot_eval], label="Pronóstico", linestyle="--")
                ax_eval.legend()
                ax_eval.set_title(f"Pronóstico vs Realidad - {var_to_plot_eval}")
                st.pyplot(fig_eval)

    # --- PASO 9: Pronóstico Futuro (Multivariante) ---
    elif current_step_num == 9:
         st.info("Objetivo: Generar pronósticos más allá de los datos disponibles y simular escenarios (Shocks).")
         
         if 'model_fit' not in st.session_state or st.session_state.model_fit is None:
             st.warning("Debes estimar el modelo en el Paso 5 primero.")
         else:
             st.subheader("Configuración del Pronóstico")
             steps_future = st.number_input("Pasos a pronosticar en el futuro:", min_value=1, value=12)
             
             if st.button("Generar Pronóstico Futuro", key="btn_future_forecast_multi"):
                 try:
                    with st.spinner("Calculando pronóstico futuro..."):
                        results = st.session_state.model_fit
                        model_type = st.session_state.model_type
                        
                        # Total steps = Future steps (we only need future, but VAR forecast returns from end of input)
                        # Actually, statsmodels VAR forecast takes 'y' (last p lags) and returns 'steps' ahead.
                        # So we just need 'steps_future'.
                        
                        # CRITICAL FIX: We need the last lags from the ENTIRE history (Train + Test), not just Train.
                        # We need to apply the SAME transformation to the Test data to get the inputs.
                        
                        # 1. Transform Test Data
                        df_test_trans = st.session_state.test_data.copy()
                        trans_type = st.session_state.transform_type
                        
                        if trans_type == "Primera Diferencia":
                            # To difference test data consistent with train, we ideally need the last train point?
                            # Actually, diff() is just row-wise. But we lose the first point of test if we just diff test.
                            # Better: Concat Train+Test, Transform, then take last lags.
                            pass
                        
                        # Construct Full Transformed Data
                        full_data = pd.concat([st.session_state.train_data, st.session_state.test_data])
                        full_data_trans = full_data.copy()
                        
                        if trans_type == "Primera Diferencia":
                            full_data_trans = full_data_trans.diff().dropna()
                        elif trans_type == "Logaritmo":
                            full_data_trans = np.log(full_data_trans).dropna()
                        elif trans_type == "Log + Diferencia":
                            full_data_trans = np.log(full_data_trans).diff().dropna()
                        elif trans_type == "Segunda Diferencia":
                            full_data_trans = full_data_trans.diff().diff().dropna()
                            
                        forecast_values_trans = None
                        
                        if model_type == "VAR":
                            lag_order = st.session_state.lags
                            # Input: Last 'lag_order' observations from the FULL transformed data
                            forecast_input = full_data_trans.values[-lag_order:]
                            
                            fc = results.forecast(y=forecast_input, steps=steps_future)
                            forecast_values_trans = pd.DataFrame(fc, columns=st.session_state.test_data.columns)
                            
                        elif model_type == "VECM":
                             # VECM predict is trickier with "future" from a specific point if not re-fitting.
                             # But statsmodels VECM 'predict' usually continues from the end of the fitted data?
                             # NO, VECM.fit() is on train data.
                             # To forecast from end of Test, we might need to re-fit or use a specific method.
                             # However, for simplicity in this lab, users often re-estimate on full data or we accept limitation.
                             # BUT, we can try to use the VAR representation of VECM to forecast from new input?
                             # Or just warn that VECM forecast is from end of Train.
                             # Let's stick to VAR fix first. For VECM, it's complex without refitting.
                             # Let's try to append test data to results? No.
                             # For now, let's just run predict, but it will likely be from end of Train.
                             # We'll add a warning for VECM if Test data exists.
                             
                             if len(st.session_state.test_data) > 0:
                                 st.warning("Nota: El modelo VECM fue ajustado con datos de entrenamiento. El pronóstico iniciará desde el final del entrenamiento, lo que puede causar un salto si el set de prueba es largo. Para VECM, se recomienda usar todo el dataset como entrenamiento si se desea pronosticar 'hoy'.")
                                 # We can't easily force VECM to start from arbitrary point without refitting.
                                 # So we just predict 'steps_future' + len(test) and take the last part?
                                 # That effectively ignores the 'reality' of the test set, but projects the model dynamics.
                                 total_steps_vecm = len(st.session_state.test_data) + steps_future
                                 fc = results.predict(steps=total_steps_vecm)
                                 forecast_values_trans = pd.DataFrame(fc, columns=st.session_state.test_data.columns)
                                 # Slice the future part
                                 forecast_values_trans = forecast_values_trans.iloc[-steps_future:].reset_index(drop=True)
                             else:
                                 fc = results.predict(steps=steps_future)
                                 forecast_values_trans = pd.DataFrame(fc, columns=st.session_state.test_data.columns)

                        # Create Date Index for Future
                        last_date = st.session_state.test_data.index[-1]
                        
                        # Robust Frequency Detection
                        freq = None
                        try:
                            if pd.api.types.is_datetime64_any_dtype(st.session_state.test_data.index):
                                freq = pd.infer_freq(st.session_state.test_data.index)
                                if freq is None:
                                     # Fallback: average diff
                                     diffs = st.session_state.test_data.index.to_series().diff().dropna()
                                     if not diffs.empty:
                                         freq = diffs.mean()
                                
                                # FIX: If freq is string, convert to offset
                                if isinstance(freq, str):
                                    freq = pd.tseries.frequencies.to_offset(freq)
                            else:
                                # Numeric index
                                freq = 1
                        except:
                            freq = 1
                        
                        # Generate future dates
                        future_dates = []
                        current_date = last_date
                        for i in range(steps_future):
                            if isinstance(freq, (pd.Timedelta, pd.DateOffset)):
                                current_date = current_date + freq
                            elif isinstance(freq, (int, float)) and (isinstance(current_date, (int, float, np.number))):
                                current_date = current_date + freq
                            elif isinstance(current_date, pd.Timestamp) and isinstance(freq, (int, float)):
                                # If timestamp but freq is number (days?), assume days? Or error?
                                # Let's assume days if not specified
                                current_date = current_date + pd.Timedelta(days=freq)
                            else:
                                # Fallback
                                # If we still have a string here (shouldn't happen with fix above) or unknown
                                try:
                                    current_date = current_date + pd.tseries.frequencies.to_offset(freq)
                                except:
                                     current_date = current_date + pd.Timedelta(days=1) # Ultimate fallback
                            future_dates.append(current_date)
                        
                        # Fix: forecast_values_trans already contains ONLY future steps (from my previous edit)
                        # So we do NOT slice it again.
                        future_fc_trans = forecast_values_trans.copy()
                        future_fc_trans.index = future_dates
                        
                        # Inverse Transform Logic (Function to reuse?)
                        def inverse_transform(df_trans, original_last_val_series, trans_type):
                            df_inv = df_trans.copy()
                            if trans_type == "Primera Diferencia":
                                for col in df_inv.columns:
                                    last_val = original_last_val_series[col]
                                    df_inv[col] = df_inv[col].cumsum() + last_val
                            elif trans_type == "Logaritmo":
                                df_inv = np.exp(df_inv)
                            elif trans_type == "Log + Diferencia":
                                 for col in df_inv.columns:
                                    last_log_val = np.log(original_last_val_series[col]) # This assumes original series is positive
                                    df_inv[col] = df_inv[col].cumsum() + last_log_val
                                 df_inv = np.exp(df_inv)
                            return df_inv

                        # For Future Forecast, we start inverse from the LAST REAL DATA POINT (End of Test)
                        last_real_vals = st.session_state.test_data.iloc[-1]
                        
                        final_future_forecast = inverse_transform(future_fc_trans, last_real_vals, st.session_state.transform_type)
                        
                        st.session_state.future_forecast = final_future_forecast
                        st.session_state.future_forecast_trans = future_fc_trans # Save transformed for shocks
                        st.success("Pronóstico futuro generado.")
                    
                 except Exception as e:
                     st.error(f"Error en Pronóstico Futuro: {e}")

             # Display Results (Persistent)
             if 'future_forecast' in st.session_state:
                 final_future_forecast = st.session_state.future_forecast
                 
                 st.subheader("Tabla de Pronósticos Futuros")
                 st.dataframe(final_future_forecast)
                 
                 st.subheader("Gráficos por Variable")
                 st.write("Línea Azul: Historia (Últimos 50 periodos). Línea Roja Punteada: Pronóstico Futuro.")
                 
                 # Combine Train + Test for History
                 history_data = pd.concat([st.session_state.train_data, st.session_state.test_data])
                 
                 for col in final_future_forecast.columns:
                     fig, ax = plt.subplots(figsize=(10, 4))
                     # Plot last 50 points of history
                     ax.plot(history_data.index[-50:], history_data[col].iloc[-50:], label="Historia")
                     ax.plot(final_future_forecast.index, final_future_forecast[col], label="Pronóstico Futuro", linestyle="--", color="red")
                     ax.set_title(f"Proyección: {col}")
                     ax.legend()
                     st.pyplot(fig)
                 
                 # --- Shock Simulation ---
                 st.markdown("---")
                 st.subheader("🌪️ Simulación de Shocks (Impulso-Respuesta)")
                 with st.expander("Configurar Shock", expanded=False):
                     st.info("Simula qué pasaría con el pronóstico si ocurre un choque inesperado en una variable.")
                     
                     shock_var = st.selectbox("Variable donde ocurre el Shock:", final_future_forecast.columns)
                     magnitude = st.number_input("Magnitud del Shock (Desviaciones Estándar):", value=1.0, step=0.5)
                     
                     if st.button("Simular Shock", key="btn_sim_shock"):
                         try:
                             results = st.session_state.model_fit
                             steps = len(final_future_forecast)
                             
                             # Get IRF
                             # irf(steps) returns steps+1 (0 to steps)
                             irf = results.irf(steps)
                             orth_irfs = irf.orth_irfs[:steps] # Slice to match forecast length
                             
                             # Indices
                             var_names = final_future_forecast.columns.tolist()
                             shock_idx = var_names.index(shock_var)
                             
                             # Calculate Shock Effect (Transformed Scale)
                             shock_effect = pd.DataFrame(index=final_future_forecast.index, columns=var_names)
                             
                             for i, col in enumerate(var_names):
                                 effect = orth_irfs[:, i, shock_idx] * magnitude
                                 shock_effect[col] = effect
                             
                             # Add to Baseline Forecast (Transformed)
                             baseline_trans = st.session_state.future_forecast_trans
                             shocked_trans = baseline_trans + shock_effect
                             
                             # Inverse Transform Shocked Forecast
                             # Reuse inverse logic (defined above or locally)
                             def inverse_transform_shock(df_trans, original_last_val_series, trans_type):
                                df_inv = df_trans.copy()
                                if trans_type == "Primera Diferencia":
                                    for col in df_inv.columns:
                                        last_val = original_last_val_series[col]
                                        df_inv[col] = df_inv[col].cumsum() + last_val
                                elif trans_type == "Logaritmo":
                                    df_inv = np.exp(df_inv)
                                elif trans_type == "Log + Diferencia":
                                     for col in df_inv.columns:
                                        last_log_val = np.log(original_last_val_series[col])
                                        df_inv[col] = df_inv[col].cumsum() + last_log_val
                                     df_inv = np.exp(df_inv)
                                return df_inv

                             last_real_vals = st.session_state.test_data.iloc[-1]
                             shocked_final = inverse_transform_shock(shocked_trans, last_real_vals, st.session_state.transform_type)
                             
                             st.session_state.shocked_forecast = shocked_final
                             st.session_state.shock_params = f"Shock en {shock_var} ({magnitude} SD)"
                             st.success("Simulación generada.")
                             
                         except Exception as e:
                             st.error(f"Error en Simulación: {e}")

                 # Display Shock Results (Persistent)
                 if 'shocked_forecast' in st.session_state:
                     st.subheader(f"Resultados: {st.session_state.shock_params}")
                     
                     shocked_df = st.session_state.shocked_forecast
                     
                     for col in shocked_df.columns:
                         fig_shock, ax_shock = plt.subplots(figsize=(10, 4))
                         ax_shock.plot(final_future_forecast.index, final_future_forecast[col], label="Pronóstico Base", linestyle="--", color="blue")
                         ax_shock.plot(shocked_df.index, shocked_df[col], label="Pronóstico con Shock", color="red")
                         ax_shock.set_title(f"Impacto en {col}")
                         ax_shock.legend()
                         st.pyplot(fig_shock)
                     

