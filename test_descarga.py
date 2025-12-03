import yfinance as yf
import pandas as pd

print("Probando descarga de datos de Yahoo Finance...\n")

tickers = ['SPY', 'QQQ', 'TSLA', 'AAPL']

for ticker in tickers:
    print(f"\n{'='*50}")
    print(f"Ticker: {ticker}")
    print('='*50)
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        
        if not data.empty:
            print(f"✅ Datos descargados: {len(data)} días")
            print(f"Columnas: {list(data.columns)}")
            print(f"\nPrimeros 3 días:")
            print(data.head(3))
            print(f"\nÚltimos 3 días:")
            print(data.tail(3))
            
            precios = data['Close']
            print(f"\nPrecio más reciente: ${precios.iloc[-1]:.2f}")
            print(f"Fecha más reciente: {precios.index[-1]}")
        else:
            print(f"❌ No se obtuvieron datos")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

print("\n" + "="*50)
print("Prueba completada")
