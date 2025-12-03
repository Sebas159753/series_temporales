with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines[:1452])

print(f"Archivo limpiado. Ahora tiene {len(lines[:1452])} líneas.")
