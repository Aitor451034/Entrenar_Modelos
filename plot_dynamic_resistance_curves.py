"""
Script para visualizar las curvas de resistencia dinámica de puntos de soldadura,
separadas por etiqueta (bueno/malo) para inspección manual.

Este script permite identificar visualmente patrones anómalos en las curvas
de resistencia que podrían indicar defectos de soldadura, complementando
el análisis del modelo de machine learning.
"""

# ==============================================================================
# 1. IMPORTACIONES DE BIBLIOTECAS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter
try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False
    print("Advertencia: mplcursors no está instalado. Instala con: pip install mplcursors")
    print("Para funcionalidad interactiva completa.")

# ==============================================================================
# 2. FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ==============================================================================

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
    Limpia y prepara el DataFrame crudo de entrada para el análisis de curvas.

    Este proceso incluye la selección de columnas relevantes, el renombramiento,
    la conversión de tipos de datos y el manejo de formatos numéricos.
    """
    # 1. Selección de columnas específicas por índice.
    # Se eligen las columnas que contienen la información necesaria para el análisis.
    # Los índices [0, 8, 9, 10, 20, 27, 67, 98] corresponden a:
    # id punto, Ns, Corrientes inst., Voltajes inst., KAI2, Ts2, Fuerza, Etiqueta datos.
    new_df = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]]

    # 2. Asignación de nombres descriptivos a las columnas seleccionadas.
    new_df.columns = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "KAI2", "Ts2", "Fuerza", "Etiqueta datos"]

    # 3. Eliminación de filas con datos faltantes críticos
    # En lugar de eliminar las últimas 2 filas automáticamente, eliminamos filas con NaN en columnas críticas
    print(f"DataFrame original: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"DataFrame después de selección de columnas: {new_df.shape[0]} filas")

    # Verificar qué filas tienen datos faltantes en columnas críticas
    cols_criticas = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "Ts2", "Etiqueta datos"]
    filas_con_nan = new_df[cols_criticas].isnull().any(axis=1)
    if filas_con_nan.any():
        print(f"Eliminando {filas_con_nan.sum()} filas con datos faltantes en columnas críticas")
        print("Filas con NaN:")
        for idx in new_df[filas_con_nan].index:
            nan_cols = new_df.loc[idx, cols_criticas].isnull()
            cols_nan = nan_cols[nan_cols].index.tolist()
            print(f"  -> Fila {idx}: NaN en {cols_nan}")
        new_df = new_df[~filas_con_nan]

    print(f"DataFrame después de limpieza: {new_df.shape[0]} filas")

    # 4. Conversión de columnas numéricas a tipo float.
    # Se utiliza pd.to_numeric con 'errors='coerce'' para convertir los valores a números.
    # Si un valor no puede convertirse (ej. es texto no numérico), se reemplaza por NaN (Not a Number).
    for col in ["KAI2", "Ts2", "Fuerza"]:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

    # 5. Redondeo de columnas de tipo float a 4 decimales.
    # Esto ayuda a estandarizar la precisión de los datos numéricos.
    float_cols = new_df.select_dtypes(include='float64').columns
    new_df = new_df.round({col: 4 for col in float_cols})

    # --- DATA CLEANING: ELIMINACIÓN DE DUPLICADOS ---
    # Se identifican filas idénticas. Se reportan los índices antes de eliminarlas.
    duplicados = new_df[new_df.duplicated()]
    if not duplicados.empty:
        print(f"\n[LIMPIEZA] Se detectaron {len(duplicados)} filas duplicadas.")
        print(f" -> Índices eliminados (originales): {duplicados.index.tolist()}")
        new_df = new_df.drop_duplicates()
    else:
        print("\n[LIMPIEZA] No se encontraron filas duplicadas.")

    # 6. Reindexación del DataFrame.
    # Se asigna un nuevo índice secuencial que comienza desde 1.
    new_df.index = range(1, len(new_df) + 1)

    # 7. Conversión general de columnas de tipo 'object' (cadenas) a float.
    # Se itera sobre todas las columnas del DataFrame original. Si una columna es de tipo 'object',
    # se intenta reemplazar las comas por puntos (formato decimal europeo a americano)
    # y luego convertir la columna completa a tipo float. Los errores se ignoran con 'try-except'.
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except:
                pass
    return new_df

def extraer_curvas_resistencia(df):
    """
    Extrae las curvas de resistencia dinámica R(t) para cada punto de soldadura.

    Para cada fila del DataFrame preprocesado, calcula la curva de resistencia
    suavizada usando el mismo procesamiento que el script principal.

    Returns:
    --------
    curvas : list of tuples
        Lista de tuplas (tiempo, resistencia_suavizada) para cada punto válido
    etiquetas : list
        Lista de etiquetas correspondientes (0=bueno, 1=malo)
    ids : list
        Lista de IDs de punto para referencia
    """
    curvas = []
    etiquetas = []
    ids = []
    count_procesados = 0
    count_invalidos = 0

    print(f"Procesando {len(df)} puntos de soldadura para extracción de curvas...")

    for idx, i in enumerate(df.index):
        try:
            id_punto = int(df.loc[i, "id punto"])
            print(f"\n[DEBUG] Procesando punto {idx+1}/4: ID {id_punto}")

            # --- 1. LECTURA DE SERIES TEMPORALES ---
            str_volt = df.loc[i, "Voltajes inst."]
            str_corr = df.loc[i, "Corrientes inst."]

            if pd.isna(str_volt) or pd.isna(str_corr):
                print(f"  -> DESCARTADO: Datos nulos (voltaje o corriente)")
                count_invalidos += 1
                continue

            # Convertir cadenas a arrays numéricos
            raw_volt = np.array([float(v) for v in str_volt.split(';') if v.strip()])
            raw_corr = np.array([float(v) for v in str_corr.split(';') if v.strip()])
            print(f"  -> Datos leídos: {len(raw_volt)} puntos voltaje, {len(raw_corr)} puntos corriente")

            # Generar eje de tiempo
            ns = int(df.loc[i, "Ns"])
            ts2 = int(df.loc[i, "Ts2"])
            t_soldadura = np.linspace(0, ts2, ns + 1)
            print(f"  -> Tiempo generado: {len(t_soldadura)} puntos, duración {ts2}ms")

            # Asegurar que todos los arrays tengan la misma longitud
            min_len = min(len(t_soldadura), len(raw_volt), len(raw_corr))
            print(f"  -> Longitud mínima: {min_len} puntos")

            # --- FILTRADO POR CANTIDAD DE DATOS ---
            if min_len < 10:
                print(f"  -> DESCARTADO: Datos insuficientes ({min_len} < 10 puntos)")
                count_invalidos += 1
                continue

            t_soldadura = t_soldadura[:min_len]
            raw_volt = raw_volt[:min_len]
            raw_corr = raw_corr[:min_len]

            # --- SIN FILTRADO DE RUIDO INICIAL ---
            # Procesamos todas las curvas con los puntos que tengan
            print(f"  -> Procesando curva completa con {len(raw_volt)} puntos")

            # --- VALIDACIÓN DE SEÑALES ---
            unique_corr = len(np.unique(raw_corr))
            unique_volt = len(np.unique(raw_volt))
            print(f"  -> Valores únicos - Corriente: {unique_corr}, Voltaje: {unique_volt}")

            if unique_corr <= 1 or unique_volt <= 1:
                print(f"  -> DESCARTADO: Señal plana (sin variación)")
                count_invalidos += 1
                continue

            print(f"  -> ✓ CURVA VÁLIDA: Procesada correctamente")

            # --- CÁLCULO DE RESISTENCIA ---
            valores_resistencia = np.divide(raw_volt, raw_corr, out=np.zeros_like(raw_volt), where=np.abs(raw_corr)>0.5)

            # --- SUAVIZADO ---
            window = min(11, len(valores_resistencia) if len(valores_resistencia)%2!=0 else len(valores_resistencia)-1)
            if window > 3:
                r_smooth = savgol_filter(valores_resistencia, window_length=window, polyorder=3)
            else:
                r_smooth = valores_resistencia

            # --- ALMACENAR RESULTADOS ---
            curvas.append((t_soldadura, r_smooth))
            etiquetas.append(int(df.loc[i, "Etiqueta datos"]))
            ids.append(int(df.loc[i, "id punto"]))
            count_procesados += 1

        except Exception as e:
            print(f"Error procesando fila {i}: {e}")
            count_invalidos += 1
            continue

    print(f"Procesamiento completado:")
    print(f"  -> Puntos válidos procesados: {count_procesados}")
    print(f"  -> Puntos inválidos/descartados: {count_invalidos}")
    print(f"  -> Total de curvas extraídas: {len(curvas)}")

    # Debug: Mostrar distribución por etiquetas
    if curvas:
        etiquetas_unicas = set(etiquetas)
        print(f"  -> Etiquetas encontradas: {sorted(etiquetas_unicas)}")
        for etiqueta in sorted(etiquetas_unicas):
            count = etiquetas.count(etiqueta)
            print(f"    -> Etiqueta {etiqueta}: {count} curvas")

    return curvas, etiquetas, ids

# ==============================================================================
# 3. FUNCIONES DE VISUALIZACIÓN
# ==============================================================================

def plot_curvas_resistencia(curvas, etiquetas, ids):
    """
    Grafica las curvas de resistencia dinámica separadas por etiqueta.

    Crea dos subplots: uno para soldaduras buenas (etiqueta 0) y otro para
    defectuosas (etiqueta 1), permitiendo la comparación visual de patrones.

    Parameters:
    -----------
    curvas : list of tuples
        Lista de tuplas (tiempo, resistencia) para cada curva
    etiquetas : list
        Etiquetas correspondientes (0=bueno, 1=malo)
    ids : list
        IDs de los puntos para referencia
    """
    if not curvas:
        print("No hay curvas válidas para graficar.")
        return

    # Separar curvas por etiqueta
    curvas_buenas = []
    curvas_malas = []
    ids_buenos = []
    ids_malos = []

    print(f"\nSeparando {len(curvas)} curvas por etiquetas...")
    for i, ((t, r), label, id_punto) in enumerate(zip(curvas, etiquetas, ids)):
        if label == 0:
            curvas_buenas.append((t, r))
            ids_buenos.append(id_punto)
        else:
            curvas_malas.append((t, r))
            ids_malos.append(id_punto)
        if i < 5:  # Mostrar primeras 5 para debug
            print(f"  -> Curva {i+1}: ID {id_punto}, Etiqueta {label}")

    print(f"Separación completada:")
    print(f"  -> Curvas buenas (0): {len(curvas_buenas)}")
    print(f"  -> Curvas malas (1): {len(curvas_malas)}")

    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # --- SUBPLOT 1: SOLDADURAS BUENAS ---
    if curvas_buenas:
        print(f"\nCurvas de soldaduras BUENAS (Etiqueta 0):")
        lines_buenas = []
        for i, (t, r) in enumerate(curvas_buenas):
            line, = ax1.plot(t, r, color='green', alpha=0.6, linewidth=1.5,
                           label=f'ID{ids_buenos[i]}')
            lines_buenas.append(line)
            print(f"  -> Curva {i+1}: ID {ids_buenos[i]}")

        ax1.set_title(f'Curvas de Resistencia Dinámica - Soldaduras BUENAS (Etiqueta 0)\n'
                     f'Total: {len(curvas_buenas)} curvas', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Tiempo (ms)', fontsize=10)
        ax1.set_ylabel('Resistencia Dinámica (Ω)', fontsize=10)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No hay curvas de soldaduras buenas',
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title('Curvas de Resistencia Dinámica - Soldaduras BUENAS (Etiqueta 0)', fontsize=12)
        lines_buenas = []

    # --- SUBPLOT 2: SOLDADURAS DEFECTUOSAS ---
    if curvas_malas:
        print(f"\nCurvas de soldaduras DEFECTUOSAS (Etiqueta 1):")
        lines_malas = []
        for i, (t, r) in enumerate(curvas_malas):
            line, = ax2.plot(t, r, color='red', alpha=0.6, linewidth=1.5,
                           label=f'ID{ids_malos[i]}')
            lines_malas.append(line)
            print(f"  -> Curva {i+1}: ID {ids_malos[i]}")

        ax2.set_title(f'Curvas de Resistencia Dinámica - Soldaduras DEFECTUOSAS (Etiqueta 1)\n'
                     f'Total: {len(curvas_malas)} curvas', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Tiempo (ms)', fontsize=10)
        ax2.set_ylabel('Resistencia Dinámica (Ω)', fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No hay curvas de soldaduras defectuosas',
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Curvas de Resistencia Dinámica - Soldaduras DEFECTUOSAS (Etiqueta 1)', fontsize=12)
        lines_malas = []

    # Configuración general
    plt.suptitle('Análisis Visual de Curvas de Resistencia Dinámica\n'
                'Inspección Manual para Detección de Anomalías', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # --- FUNCIONALIDAD INTERACTIVA ---
    if HAS_MPLCURSORS:
        # Crear cursores para ambos subplots
        cursor1 = mplcursors.cursor(ax1, hover=True)
        cursor2 = mplcursors.cursor(ax2, hover=True)

        @cursor1.connect("add")
        def on_add_cursor1(sel):
            # Obtener el índice de la línea seleccionada
            line_index = sel.artist.get_label()
            if line_index and line_index.startswith('ID'):
                sel.annotation.set_text(f'ID: {line_index[2:]}')
            else:
                # Si no tiene label, intentar encontrar el ID por posición
                x, y = sel.target
                for i, (t, r) in enumerate(curvas_buenas):
                    if len(t) > 0 and len(r) > 0:
                        # Buscar el punto más cercano
                        distances = np.sqrt((t - x)**2 + (r - y)**2)
                        min_idx = np.argmin(distances)
                        if distances[min_idx] < 0.1:  # Umbral de distancia
                            sel.annotation.set_text(f'ID: {ids_buenos[i]}')
                            break

        @cursor2.connect("add")
        def on_add_cursor2(sel):
            # Obtener el índice de la línea seleccionada
            line_index = sel.artist.get_label()
            if line_index and line_index.startswith('ID'):
                sel.annotation.set_text(f'ID: {line_index[2:]}')
            else:
                # Si no tiene label, intentar encontrar el ID por posición
                x, y = sel.target
                for i, (t, r) in enumerate(curvas_malas):
                    if len(t) > 0 and len(r) > 0:
                        # Buscar el punto más cercano
                        distances = np.sqrt((t - x)**2 + (r - y)**2)
                        min_idx = np.argmin(distances)
                        if distances[min_idx] < 0.1:  # Umbral de distancia
                            sel.annotation.set_text(f'ID: {ids_malos[i]}')
                            break

        print("\n¡Funcionalidad interactiva activada!")
        print("Pasa el mouse sobre cualquier curva para ver su ID.")
    else:
        print("\nPara funcionalidad interactiva, instala mplcursors:")
        print("pip install mplcursors")

    plt.show()

    # --- ESTADÍSTICAS ADICIONALES ---
    print("\n=== ESTADÍSTICAS DE LAS CURVAS ===")
    print(f"Total de curvas procesadas: {len(curvas)}")
    print(f"Soldaduras buenas (0): {len(curvas_buenas)}")
    print(f"Soldaduras defectuosas (1): {len(curvas_malas)}")

    if curvas_buenas:
        duraciones_buenas = [t[-1] for t, r in curvas_buenas]
        print(f"Duración media buenas: {np.mean(duraciones_buenas):.2f} ms")
        print(f"Duración máxima buenas: {np.max(duraciones_buenas):.2f} ms")

    if curvas_malas:
        duraciones_malas = [t[-1] for t, r in curvas_malas]
        print(f"Duración media malas: {np.mean(duraciones_malas):.2f} ms")
        print(f"Duración máxima malas: {np.max(duraciones_malas):.2f} ms")

# ==============================================================================
# 4. FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """
    Función principal que orquesta la carga de datos y visualización de curvas.
    """
    print("="*70)
    print("VISUALIZADOR DE CURVAS DE RESISTENCIA DINÁMICA")
    print("="*70)
    print("Este script permite inspeccionar visualmente las curvas de resistencia")
    print("de puntos de soldadura, separadas por etiqueta (bueno/malo).")
    print("Útil para identificar patrones anómalos que complementen el modelo ML.")
    print("="*70)

    # 1. Cargar archivo
    df_raw = leer_archivo()
    if df_raw is None:
        return

    # 2. Preprocesar datos
    print("\nPreprocesando datos...")
    df = preprocesar_dataframe_inicial(df_raw)

    # 3. Extraer curvas
    print("\nExtrayendo curvas de resistencia...")
    curvas, etiquetas, ids = extraer_curvas_resistencia(df)

    if not curvas:
        print("No se pudieron extraer curvas válidas. Verifica el formato del archivo.")
        return

    # 4. Visualizar
    print("\nGenerando gráficos...")
    plot_curvas_resistencia(curvas, etiquetas, ids)

    print("\n¡Visualización completada!")
    print("Revisa los gráficos para identificar patrones anómalos en las curvas.")

if __name__ == "__main__":
    main()