import pandas as pd
import tkinter as tk
from tkinter import filedialog

def leer_archivo():
    """Abre un diálogo para seleccionar un archivo CSV y lo carga como DataFrame."""
    print("Selecciona el archivo CSV que contiene los datos...")
    root = tk.Tk()
    root.withdraw()

    ruta_csv = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        filetypes=[("Archivos CSV", "*.csv")]
    )

    if not ruta_csv:
        print("Operación cancelada por el usuario.")
        return None

    try:
        df = pd.read_csv(
            ruta_csv,
            encoding="utf-8",
            sep=";",
            on_bad_lines="skip",
            header=None,
            quotechar='"',
            decimal=",",
            skiprows=3
        )
        print("¡Archivo CSV leído correctamente!")
        return df
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None
    
def preprocesar_dataframe_inicial(df):
    """
    Limpia y prepara el DataFrame crudo de entrada para la extracción de características.
    
    Este proceso incluye la selección de columnas relevantes, el renombramiento,
    la conversión de tipos de datos y el manejo de formatos numéricos.
    """
    # 1. Selección de columnas específicas por índice.
    # Se eligen las columnas que contienen la información necesaria para el análisis.
    # Los índices [0, 8, 9, 10, 20, 27, 67, 98] corresponden a:
    # id punto, Ns, Corrientes inst., Voltajes inst., KAI2, Ts2, Fuerza, Etiqueta datos.
    new_df = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]]
    
    # 2. Eliminación de las últimas dos filas.
    # Esto se hace para eliminar posibles filas de metadatos o resúmenes al final del archivo.
    new_df = new_df.iloc[:-2]
    
    # 3. Asignación de nombres descriptivos a las columnas seleccionadas.
    new_df.columns = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "KAI2", "Ts2", "Fuerza", "Etiqueta datos"]
    
    # 4. Conversión de columnas numéricas a tipo float.
    # Se utiliza pd.to_numeric con 'errors='coerce'' para convertir los valores a números.
    # Si un valor no puede convertirse (ej. es texto no numérico), se reemplaza por NaN (Not a Number).
    for col in ["KAI2", "Ts2", "Fuerza"]:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        
    # 5. Redondeo de columnas de tipo float a 4 decimales.
    # Esto ayuda a estandarizar la precisión de los datos numéricos.
    float_cols = new_df.select_dtypes(include='float64').columns
    new_df = new_df.round({col: 4 for col in float_cols})
    
    # 6. Reindexación del DataFrame.
    # Se asigna un nuevo índice secuencial que comienza desde 1.
    new_df.index = range(1, len(new_df) + 1)
    
    # 6.5. Manejo de columnas con arrays (separados por ';').
    # Algunas columnas contienen series de datos. Calculamos el promedio para tener un valor único.
    def procesar_array_string(val):
        if isinstance(val, str) and ';' in val:
            try:
                vals = [float(x.replace(',', '.').strip()) for x in val.split(';') if x.strip()]
                return sum(vals) / len(vals) if vals else None
            except:
                return None
        return val

    for col in ["Corrientes inst.", "Voltajes inst."]:
        if col in new_df.columns and new_df[col].dtype == object:
            new_df[col] = new_df[col].apply(procesar_array_string)
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

    # 7. Conversión general de columnas de tipo 'object' (cadenas) a float.
    # Se itera sobre todas las columnas del DataFrame original. Si una columna es de tipo 'object',
    # se intenta reemplazar las comas por puntos (formato decimal europeo a americano)
    # y luego convertir la columna completa a tipo float. Los errores se ignoran con 'try-except'.
    for col in new_df.columns:
        if new_df[col].dtype == object:
            try:
                new_df[col] = new_df[col].str.replace(',', '.', regex=False).astype(float)
            except:
                pass
    return new_df

def agrupar_por_id(df):
    """
    Agrupa el DataFrame por 'id punto' para facilitar la visualización.
    Calcula el promedio de las columnas numéricas y mantiene la etiqueta.
    """
    agg_dict = {
        "Ns": "mean",
        "Corrientes inst.": "mean",
        "Voltajes inst.": "mean",
        "KAI2": "mean",
        "Ts2": "mean",
        "Fuerza": "mean",
        "Etiqueta datos": "first"
    }
    
    # Filtrar el diccionario para usar solo columnas que existen en el df actual
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    if "id punto" in df.columns:
        df_agrupado = df.groupby("id punto", as_index=False).agg(agg_dict)
        return df_agrupado.round(4)
    
    print("Advertencia: No se encontró la columna 'id punto' para agrupar.")
    return df

if __name__ == "__main__":
    # 1. Cargar el archivo
    df_raw = leer_archivo()
    
    if df_raw is not None:
        # 2. Preprocesar (limpieza y selección de columnas)
        print("Procesando datos...")
        df_limpio = preprocesar_dataframe_inicial(df_raw)
        
        # 3. Agrupar por ID y mostrar resultado final
        df_final = agrupar_por_id(df_limpio)
        
        print("\n--- Resultado Final (Primeras filas) ---")
        print(df_final.head())