import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
ruta_csv = r"C:\Users\U5014554\Desktop\EntrenarModelo\DATA\Inputs_modelo_pegado_con_datos4_mas.csv"
df = pd.read_csv(ruta_csv, sep=";", low_memory=False)

# Verificar que la columna 99 existe
try:
    columna_defecto = df.columns[98]  # índice 98 = columna 99
    print(f"Usando columna: {columna_defecto}")

    # Convertir a numérico por si hay texto o valores no válidos
    df[columna_defecto] = pd.to_numeric(df[columna_defecto], errors='coerce')

    # Crear máscara: defecto si el valor es mayor a 0
    defect_mask = df[columna_defecto] > 0
    num_defect = defect_mask.sum()
    num_no_defect = (~defect_mask).sum()
    total_puntos = num_defect + num_no_defect

    # --- INICIO MODIFICACIÓN ---
    
    # 1. Mostrar en la terminal
    print("\n--- Conteo de Puntos ---")
    print(f"Puntos CON defecto: {num_defect}")
    print(f"Puntos SIN defecto: {num_no_defect}")
    print(f"Total de puntos:    {total_puntos}")
    print("------------------------\n")

    # 2. Preparar datos para el gráfico 
    labels = [f'Con defecto ({num_defect})', f'Sin defecto ({num_no_defect})']
    sizes = [num_defect, num_no_defect]
    
    # --- FIN MODIFICACIÓN ---

    colors = ['#ff6666', '#66b3ff']
    explode = (0.1, 0)

    # Crear gráfico de pastel
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct='%1.1f%%', shadow=True, startangle=140)
    
    # 3. Añadir el total al título
    plt.title(f'Distribución de puntos con y sin defecto (Total: {total_puntos})')
    plt.axis('equal')
    plt.show()

except IndexError:
    print("Error: El archivo no tiene una columna 99.")
except Exception as e:
    print(f"Ocurrió un error: {e}")