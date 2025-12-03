"""
Script de prueba para verificar que yfinance funciona correctamente
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("🔍 Probando descarga de datos con yfinance...\n")

tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'BTC-USD', 'GLD']
fecha_fin = datetime.now()
fecha_inicio = fecha_fin - timedelta(days=365)

print(f"Periodo: {fecha_inicio.date()} a {fecha_fin.date()}\n")

for ticker in tickers:
    try:
        print(f"Descargando {ticker}...", end=" ")
        data = yf.download(ticker, start=fecha_inicio, end=fecha_fin, progress=False, auto_adjust=True)
        
        if not data.empty:
            print(f"✅ OK - {len(data)} registros")
            print(f"   Columnas: {list(data.columns)}")
            
            if 'Close' in data.columns:
                precios = data['Close']
            elif isinstance(data, pd.Series):
                precios = data
            else:
                precios = data.iloc[:, 0]
            
            # Verificar el tipo de índice
            print(f"   Tipo de índice: {type(precios.index).__name__}")
            print(f"   Primera fecha: {precios.index[0]}")
            print(f"   Última fecha: {precios.index[-1]}")
            print(f"   Último precio: ${float(precios.iloc[-1]):.2f}")
            
            # Calcular volatilidad
            retornos = precios.pct_change().dropna()
            volatilidad = retornos.std() * (252 ** 0.5) * 100
            print(f"   Volatilidad anualizada: {volatilidad:.2f}%")
            print()
        else:
            print(f"❌ Sin datos")
            print()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print()

print("\n✅ Prueba completada")
