"""
Script para entrenar un modelo BalancedRandomForestClassifier con el objetivo 
de detectar puntos de soldadura defectuosos (pegados).

Este modelo maneja internamente el desbalance de clases sin necesidad de SMOTE.

El proceso incluye:
1.  Carga de datos y extracción de 32 características (feature engineering).
2.  Separación de datos en entrenamiento (Train) y prueba (Test).
3.  Definición de un pipeline de Imbalanced-learn (ImbPipeline) que:
    a. Escala los datos (StandardScaler).
    b. Realiza selección de características (RFE con RandomForest).
    c. Entrena un modelo BalancedRandomForestClassifier.
4.  Búsqueda exhaustiva de hiperparámetros (GridSearchCV) en el pipeline.
5.  Optimización del umbral de decisión basado en una precisión mínima.
6.  Evaluación final y análisis de errores en el conjunto de prueba (Test set).
7.  Guardado del pipeline completo (Scaler + Selector + Modelo) y el umbral.
"""

# ==============================================================================
# 1. IMPORTACIONES DE BIBLIOTECAS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tkinter as tk
from tkinter import filedialog

# --- Funciones científicas y estadísticas ---
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter #Filtro de Savgol
from scipy.interpolate import PchipInterpolator # Para suavizar curvas

# --- Componentes de Scikit-learn ---
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    auc, fbeta_score, make_scorer, classification_report, confusion_matrix,
    precision_score, recall_score, roc_curve, roc_auc_score
)
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier # Para usarlo como filtro en el selector
from sklearn.ensemble import RandomForestClassifier, IsolationForest # Para detección de outliers
from sklearn.impute import KNNImputer # Para imputación de valores faltantes
from sklearn.decomposition import PCA # Para visualización 2D
from sklearn.impute import SimpleImputer # Para imputación temporal en visualización

# --- NUEVAS BIBLIOTECAS: Imbalanced-learn ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ==============================================================================
# 2. CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

FEATURE_NAMES = [
    "rango_r_beta_alfa", "rango_t_e_beta", "rango_r_e_beta", "resistencia_inicial",
    "k4", "k3", "rango_intercuartilico", "desv_pre_mitad_t",
    "resistencia_ultima", "desv", "pendiente_V", "rms",
    "rango_rmax_rmin", "r_mean_post_max", "r_mean", "desv_R_pre_max",
    "pendientes_negativas_post", "rango_tiempo_max_min", "area_bajo_curva", "area_pre_mitad",
    "area_post_mitad", "max_curvatura", "num_puntos_inflexion", "max_jerk",
    "mediana", "varianza", "asimetria", "curtosis",
    "num_picos", "num_valles", "q", "m_min_cuadrados"
]

TEST_SIZE_RATIO = 0.4
RANDOM_STATE_SEED = 42
N_SPLITS_CV = 5
FBETA_BETA = 2
# Precisión mínima cambiada por el usuario
PRECISION_MINIMA = 0.73


# ==============================================================================
# 3. FUNCIONES DE CARGA Y EXTRACCIÓN DE CARACTERÍSTICAS
# ==============================================================================
# (Funciones idénticas a las versiones anteriores, colapsadas por brevedad)

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

def calcular_pendiente(resistencias, tiempos):
    """
    Calcula la pendiente (tasa de cambio) entre valores consecutivos de resistencia.
    
    Esta función aproxima la primera derivada de la resistencia con respecto al tiempo (dR/dt)
    utilizando diferencias finitas entre puntos adyacentes.
    """
    # Si no hay suficientes puntos para calcular una pendiente (0 o 1), devuelve una lista con cero.
    if len(resistencias) <= 1 or len(tiempos) <= 1:
        return [0.0]
    
    pendientes = []
    # Itera sobre cada par de puntos consecutivos (i, i+1).
    for i in range(len(resistencias) - 1):
        # Fórmula para el cambio en el tiempo (eje X): Δt = t₂ - t₁
        delta_t = tiempos[i + 1] - tiempos[i]
        # Fórmula para el cambio en la resistencia (eje Y): ΔR = R₂ - R₁
        delta_r = resistencias[i + 1] - resistencias[i]
        
        # Se previene la división por cero si dos puntos tienen el mismo tiempo.
        if delta_t == 0.0:
            # CORRECCIÓN: Si dt=0, la pendiente es infinita/indeterminada.
            # Usamos NaN para que el KNN lo impute, en vez de 0 (que significaría plano).
            pendiente_actual = np.nan
        else:
            # Fórmula de la pendiente: m = ΔR / Δt
            # Se multiplica por 100 para escalar el valor, posiblemente para expresarlo
            # en una unidad diferente o para mejorar su peso como característica en el modelo.
            pendiente_actual = (delta_r / delta_t) * 100
            
        # Eliminamos np.nan_to_num para permitir que los NaNs fluyan hasta el imputador
        pendientes.append(round(pendiente_actual, 2))
    return pendientes

def calcular_derivadas(resistencias, tiempos):
    """
    Calcula la 1ra, 2da y 3ra derivada de la curva R(t) usando np.gradient.
    
    np.gradient estima la derivada de un conjunto de puntos de datos usando
    diferencias finitas centrales para los puntos interiores y diferencias de 
    primer orden (hacia adelante/atrás) en los bordes. Es más preciso que
    calcular la pendiente simple entre dos puntos.
    """
    # Si no hay suficientes puntos, no se puede calcular la derivada.
    if len(resistencias) <= 1 or len(tiempos) <= 1:
        # Devuelve arrays de NaNs si no hay datos suficientes
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
        
    # --- 1ª Derivada (Velocidad de cambio de la resistencia) ---
    # Fórmula (simplificada para puntos interiores): f'(x) ≈ (f(x+h) - f(x-h)) / 2h
    # np.gradient(y, x) calcula la derivada dy/dx.
    # Aquí, calcula dR/dt, que representa la "velocidad" de cambio de la resistencia.
    primera_derivada = np.gradient(resistencias, tiempos)
    
    # --- 2ª Derivada (Aceleración de la resistencia) ---
    # Se calcula la derivada de la primera derivada para obtener la segunda (d²R/dt²).
    # Representa la "aceleración" o la concavidad/curvatura de la curva R(t).
    segunda_derivada = np.gradient(primera_derivada, tiempos)
    
    # --- 3ª Derivada (Jerk o "sacudida" de la resistencia) ---
    # Se calcula la derivada de la segunda derivada para obtener la tercera (d³R/dt³).
    # Representa el "jerk" o la tasa de cambio de la aceleración.
    tercera_derivada = np.gradient(segunda_derivada, tiempos)
    
    # CORRECCIÓN: Devolvemos las derivadas tal cual (con NaNs si existen).
    # Razón: Si la señal es plana, np.gradient ya devuelve 0.0 correctamente.
    # Si devuelve NaN, es un dato faltante que el KNNImputer del pipeline debe rellenar.
    # Forzar a 0 confundiría "falta de datos" con "estabilidad perfecta".
    
    # Nota: Los 'Inf' (infinitos) se convertirán a NaN en la limpieza final de 'extraer_features'.
    return primera_derivada, segunda_derivada, tercera_derivada

def preprocesar_dataframe_inicial(df):
    """
    Limpia y prepara el DataFrame crudo de entrada para la extracción de características.
    
    Este proceso incluye la selección de columnas relevantes, el renombramiento,
    la conversión de tipos de datos y el manejo de formatos numéricos.
    """
    # 1. Selección de columnas específicas por índice.
    # Se eligen las columnas que contienen la información necesaria para el análisis.
    # Los índices [0, 8, 9, 10, 20, 27, 67, 98] corresponden a:
    
    # [ROBUSTEZ] Verificamos que el DF tenga suficientes columnas antes de hacer iloc
    if df.shape[1] < 99:
        print(f"[ERROR CRÍTICO] El CSV tiene {df.shape[1]} columnas, se esperaban al menos 99.")
        return None

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

def extraer_features_fila_por_fila(new_df):
    """
    Versión OPTIMIZADA Y CORREGIDA.
    Calcula las 32 características físicas y estadísticas filtrando el ruido.
    
    Esta función itera sobre cada punto de soldadura (fila) del DataFrame 
    preprocesado y extrae un vector de 32 características numéricas que 
    describen la curva de resistencia dinámica (R(t)) y otras señales.
    
    El proceso para cada punto es:
    1. Leer las series temporales de voltaje y corriente.
    2. Calcular la resistencia dinámica R(t) = V(t) / I(t).
    3. Suavizar la curva de resistencia para eliminar ruido usando un filtro Savitzky-Golay.
    4. Identificar puntos clave en la curva R(t) suavizada (inicio, fin, máximo, mínimo).
    5. Calcular la energía total de la soldadura (integral de la potencia).
    6. Calcular derivadas (velocidad, aceleración, jerk) de la curva R(t).
    7. Extraer características geométricas, estadísticas y físicas de estas curvas.
    8. Ensamblar las 32 características en un único vector.
    """
    X_calculado = []
    y_calculado = []
    
    # Contadores para reporte de limpieza
    count_insufficient = 0

    print(f"Procesando {len(new_df)} puntos de soldadura (Algoritmo Corregido)...")

    for i in new_df.index:
        try:
            # --- 1. LECTURA Y LIMPIEZA DE SERIES ---
            # Se leen las cadenas de texto que contienen los valores de voltaje y corriente,
            # separados por ';'.
            str_volt = new_df.loc[i, "Voltajes inst."]
            str_corr = new_df.loc[i, "Corrientes inst."]
            
            if pd.isna(str_volt) or pd.isna(str_corr):
                print(f"Fila {i}: Datos nulos. Saltando.")
                continue
                
            # Se convierten las cadenas a arrays de números (float).
            raw_volt = np.array([float(v) for v in str_volt.split(';') if v.strip()])
            raw_corr = np.array([float(v) for v in str_corr.split(';') if v.strip()])
            
            # Se genera el eje de tiempo para la soldadura.
            # Ns: Número de muestras.
            # Ts2: Tiempo total de soldadura en milisegundos.
            ns = int(new_df.loc[i, "Ns"])
            ts2 = int(new_df.loc[i, "Ts2"])
            t_soldadura = np.linspace(0, ts2, ns + 1)
            
            # Se asegura que todos los arrays (tiempo, voltaje, corriente) tengan la misma longitud
            # para evitar errores en cálculos vectoriales.
            min_len = min(len(t_soldadura), len(raw_volt), len(raw_corr))
            
            # --- DATA CLEANING: FILTRADO POR CANTIDAD DE DATOS ---#
            if min_len <= 10: 
                print(f"[LIMPIEZA] Fila {i} ELIMINADA: Datos insuficientes ({min_len} puntos).")
                count_insufficient += 1
                continue 
            
            t_soldadura = t_soldadura[:min_len]
            raw_volt = raw_volt[:min_len]
            raw_corr = raw_corr[:min_len]

            # --- DATA CLEANING: ELIMINACIÓN DE PUNTOS INICIALES ---
            # Se busca el primer índice donde la INTENSIDAD sea válida.
            # Solo nos fiamos de la corriente para decidir si ha empezado la soldadura.
            start_idx = 0
            
            # Usamos solo raw_corr para detectar el inicio real del proceso
            while start_idx < len(raw_corr) and raw_corr[start_idx] < 100:
                start_idx += 1
            
            # Comprobación de seguridad por si toda la soldadura es ruido bajo
            if start_idx >= len(raw_volt):
                print(f"[LIMPIEZA] Fila {i} ELIMINADA: Señal completa por debajo de 100.")
                count_insufficient += 1
                continue

            if start_idx > 0:
                # [DEBUG] Imprimir mensaje para confirmar que el recorte está ocurriendo
                print(f"[INFO] Fila {i}: Recortando {start_idx} puntos iniciales (ruido < 100).")

                # Recortar señales
                raw_volt = raw_volt[start_idx:]
                raw_corr = raw_corr[start_idx:]
                
                # Recortar y resetear tiempo (Mantiene el 'dt' original y pone el inicio en 0)
                t_soldadura = t_soldadura[start_idx:]
                t_soldadura = t_soldadura - t_soldadura[0] 
                
                # Actualizar Ts2 (feature de duración) al nuevo tiempo efectivo
                ts2 = t_soldadura[-1]
                
                # Verificación final de longitud, punto con un tiempo menor a 25 ms se descarta por los pocos puntos que presenta, menor a 10.
                if len(raw_volt) <= 10:
                    print(f"[LIMPIEZA] Fila {i} ELIMINADA: Datos insuficientes tras recorte (<10).")
                    count_insufficient += 1
                    continue
            
            # Si Intensidad o Voltaje tienen un único valor (varianza 0), eliminamos la fila.
            # Nota: count_flat_signal no estaba inicializado en el código original, lo corregimos a un print
            if len(np.unique(raw_corr)) <= 1 or len(np.unique(raw_volt)) <= 1:
                print(f"[LIMPIEZA] Fila {i} ELIMINADA: Señal plana (varianza 0).")
                continue
            
            # --- 2. CÁLCULO DE RESISTENCIA (FILTRADO) ---
            # Se calcula la resistencia dinámica usando la Ley de Ohm: R(t) = V(t) / I(t).
            # Se usa np.divide con 'where' para evitar la división por cero si la corriente es muy baja.
            valores_resistencia = np.divide(raw_volt, raw_corr, out=np.zeros_like(raw_volt), where=np.abs(raw_corr)>0.5)
            
            # Se aplica un filtro Savitzky-Golay a la señal de resistencia.
            # Este filtro suaviza la curva ajustando un polinomio a subconjuntos de datos,
            # lo que es crucial para obtener derivadas estables y reducir el ruido del sensor.
            window = min(11, len(valores_resistencia) if len(valores_resistencia)%2!=0 else len(valores_resistencia)-1)
            if window > 3:
                r_smooth = savgol_filter(valores_resistencia, window_length=window, polyorder=3)
            else:
                r_smooth = valores_resistencia

            # --- 3. EXTRACCIÓN DE PUNTOS CLAVE ---
            # Se identifican los puntos más importantes de la curva de resistencia suavizada.
            
            # Identificar pico Beta real (calentamiento) tras la caída inicial.
            valles_temp, _ = find_peaks(-r_smooth)
            if len(valles_temp) > 0:
                idx_valley_start = valles_temp[0] # Primer valle (fin caída contacto)
            else:
                idx_valley_start = 0
            
            # Buscar máximo (Beta) solo después del valle inicial
            if idx_valley_start < len(r_smooth) - 1:
                idx_max_rel = np.argmax(r_smooth[idx_valley_start:])
                idx_max = idx_valley_start + idx_max_rel
            else:
                idx_max = np.argmax(r_smooth)

            idx_min = np.argmin(r_smooth) # Índice del valor mínimo de resistencia.
            
            resistencia_max = r_smooth[idx_max] # Valor máximo de resistencia (Beta).
            t_R_max = t_soldadura[idx_max]      # Tiempo en el que ocurre el máximo.
            
            # Valores en el valle (inicio de la subida K3)
            r_valley_start = r_smooth[idx_valley_start]
            t_valley_start = t_soldadura[idx_valley_start]
            
            r0 = r_smooth[0]      # Resistencia inicial (Alfa).
            r_e = r_smooth[-1]    # Resistencia final.
            t_e = t_soldadura[-1] # Tiempo final.
            resistencia_min = np.min(r_smooth) # Valor mínimo de resistencia.
            t_min = t_soldadura[idx_min]       # Tiempo en el que ocurre el mínimo.

            # --- 4. CÁLCULO DE ENERGÍA ---
            # Se calcula la energía total disipada durante la soldadura en Joules.
            # Fórmula: Energía (Q) = ∫ P(t) dt = ∫ V(t) * I(t) dt
            # Se convierten las unidades a estándar (Amperios, Voltios, Segundos).
            i_amperios = raw_corr * 1000.0      # Corriente de kA a A.
            v_reales = raw_volt / 100.0         # Voltaje (ajustar según la escala del sensor).
            t_segundos = t_soldadura / 1000.0   # Tiempo de ms a s.
            
            potencia = v_reales * i_amperios # Potencia instantánea P(t) = V(t) * I(t).
            # Se integra la potencia respecto al tiempo usando la regla del trapecio.
            q_joules = np.trapezoid(potencia, x=t_segundos)

            # --- 5. DERIVADAS Y EVENTOS FÍSICOS ---
            # Se calculan las derivadas de la curva de resistencia suavizada para analizar su dinámica.
            # d1 (1ª derivada, dR/dt): Velocidad de cambio de la resistencia.
            # d2 (2ª derivada, d²R/dt²): Aceleración de la resistencia (concavidad/curvatura).
            # d3 (3ª derivada, d³R/dt³): Jerk o sobreaceleración de la resistencia.
            d1 = np.gradient(r_smooth, t_soldadura)
            d2 = np.gradient(d1, t_soldadura)
            d3 = np.gradient(d2, t_soldadura)
            
            # Característica 22: Máxima curvatura de la señal R(t).
            max_curvatura = np.nanmax(np.abs(d2)) if not np.all(np.isnan(d2)) else np.nan
            # Característica 24: Máximo jerk de la señal R(t).
            max_jerk = np.nanmax(np.abs(d3)) if not np.all(np.isnan(d3)) else np.nan
            # Característica 23: Número de puntos de inflexión. Se cuentan los cruces por cero de la 2ª derivada.
            puntos_inflexion = np.sum(np.diff(np.sign(d2)) != 0)
            
            # Se cuentan los picos (máximos locales) y valles (mínimos locales) en la curva R(t).
            picos, _ = find_peaks(r_smooth)
            valles, _ = find_peaks(-r_smooth)
            # Característica 29: Número de picos.
            num_picos = len(picos)
            # Característica 30: Número de valles.
            num_valles = len(valles)

            # --- 6. CÁLCULOS ESTADÍSTICOS Y PENDIENTES ---
            
            # Característica 32: Pendiente de la recta de regresión por mínimos cuadrados (OLS).
            # Representa la tendencia general de la resistencia durante todo el proceso.
            # Fórmula: m = Cov(R, t) / Var(t)
            t_mean = np.mean(t_soldadura)
            r_mean = np.mean(r_smooth)
            numerador = np.sum((r_smooth - r_mean) * (t_soldadura - t_mean))
            denominador = np.sum((t_soldadura - t_mean)**2) 
            # Si denominador es 0 (solo 1 punto de tiempo), la pendiente es indeterminada -> NaN
            m_ols = numerador / denominador if denominador != 0 else np.nan

            # Característica 11: Pendiente de la curva de voltaje desde su pico hasta el final.
            idx_v_max = np.argmax(raw_volt)
            
            if idx_v_max < len(raw_volt) - 1:
                dt_v = t_soldadura[idx_v_max] - t_e
                if dt_v != 0:
                     pendiente_V = (raw_volt[idx_v_max] - raw_volt[-1]) / dt_v
                else:
                     pendiente_V = 0.0
            else:
                # Si el pico está al final, no hubo caída de voltaje (posible defecto).
                # Asignamos 0.0 para indicar "sin pendiente" en lugar de NaN.
                pendiente_V = 0.0

            # Característica 6 (k3): Pendiente de subida (calentamiento).
            # Se calcula desde el valle inicial (R_alfa efectivo) hasta el pico Beta.
            # REGLA FÍSICA: t_beta - t_alfa > 0. Si es negativo, es un error de cálculo -> NaN.
            # Para este caso el valor de pendiente calculado tiene que ser positivo
            dt_rise = t_R_max - t_valley_start
            if dt_rise > 0:
                k3 = (resistencia_max - r_valley_start) / dt_rise * 100
                # Si k3 <= 0, significa que no hubo subida real (posible defecto).
                # Mantenemos el valor para que el modelo detecte la anomalía.
            else:
                # Si no hay tiempo de subida (pico antes o en el valle), no hubo fase de calentamiento.
                # Asignamos 0.0 para indicar ausencia de pendiente.
                k3 = 0.0
            
            # Característica 5 (k4): Pendiente desde el pico de resistencia hasta el final.
            # Mide la velocidad de enfriamiento o colapso. Fórmula: (R_final - R_max) / (t_final - t_R_max)
            # REGLA FÍSICA: t_e - t_beta > 0. Si es negativo, es un error de cálculo -> NaN
            # El valor de pendiente calculado debe ser negativo
            delta_t_post = t_e - t_R_max
            if delta_t_post > 0:
                k4 = ((r_e - resistencia_max) / delta_t_post * 100)
                # [CORRECCIÓN] Si k4 >= 0 (pendiente positiva/plana), NO poner NaN.
                # Un valor positivo indica que NO hubo enfriamiento correcto (posible defecto).
                # Dejar el valor real permite al modelo aprender que "k4 >= 0" -> "Defecto".
                # Si usamos KNN aquí, disfrazaríamos el defecto con valores de soldaduras sanas.
            else:
                # Si el pico está al final, no hubo fase de enfriamiento observable.
                # Asignamos 0.0 (plano) para indicar explícitamente esta anomalía física.
                k4 = 0.0

            # Característica 17: Número de puntos con pendiente negativa después del pico de resistencia.
            pendientes_post = d1[idx_max:]
            num_negativas = np.sum(pendientes_post < 0)

            # --- Características Estadísticas sobre la curva R(t) ---
            # Característica 10: Desviación estándar de toda la curva de resistencia.
            desv = np.std(r_smooth)
            # Característica 12: Valor eficaz (Root Mean Square) de la resistencia. Mide la "potencia" de la señal.
            rms = np.sqrt(np.mean(r_smooth**2))
            # Característica 25: Mediana de la resistencia.
            mediana = np.median(r_smooth)
            # Característica 26: Varianza de la resistencia.
            varianza = np.var(r_smooth)
            # Característica 7: Rango intercuartílico (IQR). Mide la dispersión del 50% central de los datos.
            iqr = np.percentile(r_smooth, 75) - np.percentile(r_smooth, 25)
            # Característica 27: Coeficiente de asimetría (Skewness).
            asim = skew(r_smooth) if len(r_smooth) > 2 else np.nan
            # Característica 28: Curtosis. Mide qué tan "puntiaguda" es la distribución.
            curt = kurtosis(r_smooth) if len(r_smooth) > 2 else np.nan
            
            # --- Características Estadísticas Parciales ---
            # Característica 8: Desviación estándar de la primera mitad temporal de la curva.
            desv_pre_mitad_t = np.std(r_smooth[:len(r_smooth)//2])
            # Característica 16: Desviación estándar de la resistencia antes del pico.
            desv_R = np.std(r_smooth[:idx_max+1])
            # Característica 14: Resistencia media después del pico.
            r_mean_post_max = np.mean(r_smooth[idx_max:]) if idx_max < len(r_smooth) else np.nan

            # --- Características basadas en Áreas ---
            # Se calcula el área bajo la curva de R(t) usando la regla del trapecio.
            # Característica 19: Área total bajo la curva R(t).
            area_total = np.trapezoid(r_smooth, t_soldadura)
            idx_mitad = len(t_soldadura) // 2
            # Característica 20: Área en la primera mitad del tiempo.
            area_pre_mitad = np.trapezoid(r_smooth[:idx_mitad], t_soldadura[:idx_mitad])
            # Característica 21: Área en la segunda mitad del tiempo.
            area_post_mitad = area_total - area_pre_mitad
            
            # --- Características basadas en Rangos ---
            # Característica 1: Rango de resistencia entre el pico y el inicio (R_max - R_inicial).
            rango_r_beta_alfa = resistencia_max - r0
            # Característica 3: Rango de resistencia entre el final y el pico (R_final - R_max).
            rango_r_e_beta = r_e - resistencia_max
            # Característica 2: Rango de tiempo entre el final y el pico (t_final - t_R_max).
            rango_t_e_beta = t_e - t_R_max
            # Característica 13: Rango entre la resistencia máxima y mínima.
            rango_rmax_rmin = resistencia_max - resistencia_min
            # Característica 18: Rango de tiempo entre el máximo y el mínimo.
            rango_tiempo_max_min = t_R_max - t_min

            # --- 7. ENSAMBLAJE FINAL (ORDEN DE TU CÓDIGO ORIGINAL) ---
            # Se recopilan todas las características calculadas en una lista, en un orden específico
            # que coincide con la lista `FEATURE_NAMES`.
            fila_features = [
                float(rango_r_beta_alfa),       # 1. rango_r_beta_alfa: R_max - R_inicial
                float(rango_t_e_beta),          # 2. rango_t_e_beta: t_final - t_R_max
                float(rango_r_e_beta),          # 3. rango_r_e_beta: R_final - R_max
                float(r0),                      # 4. resistencia_inicial: R al tiempo t=0
                float(k4),                      # 5. k4: Pendiente de R(t) post-pico
                float(k3),                      # 6. k3: Pendiente de R(t) pre-pico
                float(iqr),                     # 7. rango_intercuartilico: IQR de R(t)
                float(desv_pre_mitad_t),        # 8. desv_pre_mitad_t: Desv. estándar de R(t) en la primera mitad del tiempo
                float(r_e),                     # 9. resistencia_ultima: R al tiempo t=final
                float(desv),                    # 10. desv: Desv. estándar de toda la curva R(t)
                float(pendiente_V),             # 11. pendiente_V: Pendiente de V(t) desde su pico hasta el final
                float(rms),                     # 12. rms: Valor eficaz (RMS) de R(t)
                float(rango_rmax_rmin),         # 13. rango_rmax_rmin: R_max - R_min
                float(r_mean_post_max),         # 14. r_mean_post_max: Media de R(t) después del pico
                float(r_mean),                  # 15. r_mean: Media de toda la curva R(t)
                float(desv_R),                  # 16. desv_R_pre_max: Desv. estándar de R(t) antes del pico
                float(num_negativas),           # 17. pendientes_negativas_post: Conteo de pendientes < 0 después del pico
                float(rango_tiempo_max_min),    # 18. rango_tiempo_max_min: t_R_max - t_R_min
                float(area_total),              # 19. area_bajo_curva: Integral de R(t) dt
                float(area_pre_mitad),          # 20. area_pre_mitad: Integral de R(t) en la primera mitad del tiempo
                float(area_post_mitad),         # 21. area_post_mitad: Integral de R(t) en la segunda mitad del tiempo
                float(max_curvatura),           # 22. max_curvatura: Máximo de la 2ª derivada de R(t)
                float(puntos_inflexion),        # 23. num_puntos_inflexion: Conteo de cruces por cero de la 2ª derivada
                float(max_jerk),                # 24. max_jerk: Máximo de la 3ª derivada de R(t)
                float(mediana),                 # 25. mediana: Mediana de R(t)
                float(varianza),                # 26. varianza: Varianza de R(t)
                float(asim),                    # 27. asimetria: Asimetría (skewness) de R(t)
                float(curt),                    # 28. curtosis: Curtosis de R(t)
                float(num_picos),               # 29. num_picos: Número de máximos locales en R(t)
                float(num_valles),              # 30. num_valles: Número de mínimos locales en R(t)
                float(q_joules),                # 31. q: Energía total de la soldadura en Joules (Integral de V*I dt)
                float(m_ols)                    # 32. m_min_cuadrados: Pendiente de la regresión lineal de R(t)
            ]
            
            # --- DETECCIÓN DE NaNs ---
            # Imprimir por pantalla qué características tienen NaN antes de sustituir
            arr_temp = np.array(fila_features)
            if np.isnan(arr_temp).any():
                indices_nan = np.where(np.isnan(arr_temp))[0]
                print(f"\n[AVISO] Fila {i}: Se encontraron valores NaN en las siguientes características:")
                for idx in indices_nan:
                    print(f"   -> {FEATURE_NAMES[idx]}")

            # Limpieza final: Convertir infinitos a NaN y MANTENER los NaNs para el imputador
            # NOTA: Ya NO reemplazamos NaNs por 0.0 aquí. 
            # Se mantendrán como NaN para ser imputados por KNN en el pipeline.
            fila_features = np.array(fila_features)
            fila_features[np.isinf(fila_features)] = np.nan
            
            X_calculado.append(fila_features)
            y_calculado.append(int(new_df.loc[i, "Etiqueta datos"]))

        except Exception as e:
            print(f"Error en fila {i}: {e}")
            continue

    # --- REPORTE FINAL DE LIMPIEZA ---
    if count_insufficient == 0:
        print("[LIMPIEZA] No se encontraron filas con datos insuficientes.")

    print("Cálculo de features completado.")
    return np.array(X_calculado), np.array(y_calculado)


# ==============================================================================
# 4. FUNCIONES DEL PIPELINE DE MACHINE LEARNING
# ==============================================================================

def graficar_distribucion_energia(X, y):
    """Diagnóstico: Grafica histograma de la Energía (q) para ver separación de clases."""
    print("\n--- Generando Gráfico de Diagnóstico de Energía (q) ---")
    plt.figure(figsize=(10, 6))
    
    # Intentamos encontrar la columna 'q' por nombre si X es DataFrame, o índice 30
    try:
        if isinstance(X, pd.DataFrame):
            col_q = X['q']
        else:
            col_q = X[:, 30] 
        
        q_buenos = col_q[y == 0]
        q_defectos = col_q[y == 1]
        
        sns.histplot(q_buenos, color='green', label='OK (Sin Defecto)', kde=True, element="step")
        sns.histplot(q_defectos, color='red', label='No OK (Defecto)', kde=True, element="step")
        
        plt.title('DIAGNÓSTICO: Distribución de Energía Real (Joules) por Clase')
        plt.xlabel('Energía (q) - Joules Calculados con V*I')
        plt.legend()
        plt.show()
        print("Gráfico generado. Si las curvas están muy separadas, es NORMAL tener un score alto.")
    except Exception as e:
        print(f"No se pudo generar el gráfico de energía: {e}")

def visualizar_datos_previos(X, y, feature_names):
    """
    Visualiza la distribución de los datos mediante PCA (2D) y Boxplots.
    Permite identificar visualmente outliers y la separabilidad de las clases
    ANTES de tomar decisiones de eliminación.
    """
    print("\n" + "="*70)
    print("PASO 1.5: VISUALIZACIÓN DE DATOS (PCA y Boxplots)")
    print("="*70)

    # 1. Estandarización temporal para visualización (PCA y Boxplots requieren misma escala)
    scaler = StandardScaler()
    # Manejo de NaNs para visualización: Imputación simple temporal (media)
    # (Solo para ver los gráficos, no afecta al dataset real que usará KNN después)
    imputer = SimpleImputer(strategy='mean') 
    
    # Pipeline temporal de visualización
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # --- GRÁFICO 1: PCA 2D (Mapa de puntos) ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    # Clase 0: Sin Defecto (Verde)
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='green', label='Sin Defecto (0)', alpha=0.5, edgecolors='k', s=50)
    # Clase 1: Con Defecto (Rojo)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', label='Con Defecto (1)', alpha=0.7, edgecolors='k', marker='^', s=70)
    
    plt.title(f'Mapa de Distribución PCA (2D)\nVarianza explicada: {sum(pca.explained_variance_ratio_):.2%}')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
    
    # --- GRÁFICO 2: BOXPLOTS (Detectar Outliers por variable) ---
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=df_scaled, orient='h', palette="Set2")
    plt.title('Distribución de Características Estandarizadas (Z-Scores)')
    plt.xlabel('Desviaciones Estándar (Puntos > 3 o < -3 suelen ser outliers)')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # --- ANÁLISIS DE ANOMALÍAS (ISOLATION FOREST) ---
    # ¿Cómo detectar outliers sin borrar datos válidos de otros espesores?
    # Isolation Forest aísla puntos "raros". Si tienes varios espesores (clusters), 
    # el algoritmo suele respetarlos y solo marca lo que está lejos de CUALQUIER cluster.
    print("\n--- Análisis de Posibles Outliers (Informativo) ---")
    print("Identificando los puntos más atípicos matemáticamente (posibles errores de sensor).")
    print("NOTA: No se eliminarán filas, solo se listan para tu revisión manual.")
    
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_imputed)
    scores = iso.decision_function(X_imputed) # Menor score = Más anómalo
    
    # Crear un DataFrame temporal para mostrar resultados
    df_outliers = pd.DataFrame({'Indice_Fila': X.index, 'Clase': y.values, 'Score_Anomalia': scores})
    
    # Mostrar los 5 puntos más "raros" de cada clase
    for clase_val, nombre in [(0, "Sin Defecto (OK)"), (1, "Con Defecto (NOK)")]:
        print(f"\nTop 5 puntos más anómalos en clase '{nombre}':")
        # Ordenamos por score ascendente (los más negativos son los más anómalos)
        top_anomalos = df_outliers[df_outliers['Clase'] == clase_val].sort_values('Score_Anomalia').head(5)
        for _, row in top_anomalos.iterrows():
            print(f"  -> Fila {int(row['Indice_Fila'])} | Score: {row['Score_Anomalia']:.4f}")

    print("✓ Gráficos generados. Revisa si hay puntos muy alejados en el PCA o los Boxplots.")

def paso_1_cargar_y_preparar_datos(feature_names):
    """Orquesta la carga de datos y la creación de los DataFrames X e y."""
    df_raw = leer_archivo()
    if df_raw is None:
        return None, None
    df_preprocesado = preprocesar_dataframe_inicial(df_raw)
    if df_preprocesado is None:
        return None, None
        
    X_raw, y_raw = extraer_features_fila_por_fila(df_preprocesado)
    
    if X_raw.size == 0:
        print("No se cargaron datos. Terminando.")
        return None, None
        
    X = pd.DataFrame(X_raw, columns=feature_names)
    X = X.applymap(lambda x: round(x, 4))
    y = pd.Series(y_raw, name="Etiqueta_Defecto")

    print("\n" + "="*70)
    print("TABLA DE VALORES DE CARACTERÍSTICAS (TODAS LAS FILAS)")
    print("="*70)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
        print(X)
    
    print("\n" + "="*70)
    print("VERIFICACIÓN DE NULOS (NaNs)")
    print("="*70)
    nulos = X.isnull().sum()
    if nulos.sum() == 0:
        print("¡INCREÍBLE! No hay ningún NaN en todo el dataset.")
        print("La limpieza previa eliminó todas las filas problemáticas.")
        print("El KNN actuará solo como precaución.")
    else:
        print("Se encontraron los siguientes NaNs que el KNN arreglará después:")
        print(nulos[nulos > 0])

    print("\n--- Resumen de Datos Cargados ---")
    print(f"Total de muestras: {len(X)}")
    print(f"Distribución de clases:\n{y.value_counts(normalize=True)}")
    print("----------------------------------\n")
    return X, y

def paso_2_escalar_y_dividir_datos(X, y, test_size, random_state):
    """
    Solo divide los datos. El escalado y selección pasan al pipeline del Paso 3.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    print("Datos divididos en Train y Test (sin escalar ni filtrar aún).")
    
    return X_train, X_test, y_train, y_test

def paso_3_entrenar_modelo(X_train, y_train, n_splits, fbeta, random_state):
    """
    *** LÓGICA CENTRAL: Pipeline Completo (Scaler + Selector + Balanced RF + KNN) ***
    """
    print("Iniciando búsqueda de hiperparámetros para Balanced RandomForest...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f2_scorer = make_scorer(fbeta_score, beta=fbeta)

    # 1. Definir el Pipeline
    # NO necesitamos definir la variable modelo_hibrido_BRF fuera, 
    # lo instanciamos directamente dentro del pipeline para evitar confusiones.
    
    pipeline_BRF = ImbPipeline([
        ('scaler', StandardScaler()),           # 1. Escalar
        
        # Imputación de valores faltantes (KNN)
        ('imputer', KNNImputer(weights='uniform')),
        
        ('selector', RFE(           # 2. Seleccionar Features
            # RFE con RandomForest es ideal para inputs numéricos y target categórico.
            # El árbol encuentra "cortes" (thresholds) óptimos en los valores numéricos
            # para separar las clases, capturando relaciones no lineales que otros no verian.
            RandomForestClassifier(n_estimators=400, random_state=random_state, n_jobs=-1,),
            step=0.1,  # Elimina el 10% de features en cada paso (más rápido que 1 a 1)
            verbose=0
        )),
        ('model', BalancedRandomForestClassifier( # 3. Modelo Final (El esqueleto)
            random_state=random_state,
            sampling_strategy="auto",
            replacement=False,
            n_jobs=-1
            # NOTA: Aquí no ponemos max_depth, porque eso lo decide el Grid abajo
        ))
    ])

    # 2. Definir el GRID (Aquí están los controles anti-sobreajuste)
    param_grid_BRF = {
        # --- Parámetros del Imputador KNN ---
        'imputer__n_neighbors': [3, 5, 7],   # El modelo probará cuál es el mejor valor
        # --- Balanced Random Forest (Anti-Overfitting) ---
        "model__n_estimators": [200, 300],      # Bastantes árboles para estabilidad
        "model__max_depth": [4, 6, 8],          # <--- ESTO evita el sobreajuste (profundidad baja)
        "model__min_samples_leaf": [5, 10],     # <--- ESTO obliga a generalizar (grupos grandes)
        "model__max_features": ["sqrt","log2"],        # <--- ESTO reduce la varianza
        "model__class_weight": ["balanced", "balanced_subsample"],
        'selector__n_features_to_select': [15 ,20 ,25]              # --- Parámetros del Selector ---#
    }
    
    total_combinaciones = np.prod([len(v) for v in param_grid_BRF.values()])
    print(f"GridSearchCV probará {total_combinaciones} combinaciones.")
    print("Entrenando... (Esto puede tardar)")

    # 3. Ejecutar GridSearch
    search_cv = GridSearchCV(
        estimator=pipeline_BRF,
        param_grid=param_grid_BRF,
        cv=skf,
        scoring=f2_scorer,
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    search_cv.fit(X_train, y_train)
    
    mejor_modelo = search_cv.best_estimator_
    print("Entrenamiento completado.")
    print(f"Mejores parámetros: {search_cv.best_params_}")
    print(f"Mejor score F2 (en CV): {search_cv.best_score_:.4f}")
    
    return mejor_modelo

def paso_4_evaluar_importancia_y_umbral_defecto(mejor_modelo, X_test, y_test, feature_names):
    """
    Grafica la importancia de características y la matriz de confusión
    con el umbral por defecto (0.5).
    """
   # --- 1. Recuperar nombres correctos tras la selección ---
    selector = mejor_modelo.named_steps['selector']
    mask = selector.get_support()
    
    # Filtramos los nombres de las 32 columnas originales
    nombres_finales = np.array(feature_names)[mask]
    
    # Obtenemos importancias del modelo
    importancias = mejor_modelo.named_steps['model'].feature_importances_

    # --- 2. Crear DataFrame ---
    df_importancias = pd.DataFrame({
        'predictor': nombres_finales,
        'importancia': importancias
    }).sort_values(by='importancia', ascending=True)

    print(f"\nImportancia de las {len(df_importancias)} características seleccionadas:")
    print(df_importancias.sort_values(by='importancia', ascending=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_importancias.predictor, df_importancias.importancia)
    plt.yticks(size=8)
    ax.set_xlabel('Importancia de la Característica')
    ax.set_ylabel('Variable Predictora')
    # *** CORRECCIÓN ***: Título del gráfico
    ax.set_title('Importancia de Características (Balanced RandomForest)')
    plt.tight_layout()
    plt.show()

    # --- 2. Matriz de Confusión (Umbral 0.5) ---
    predicciones_defecto = mejor_modelo.predict(X_test)
    matriz_confusion = confusion_matrix(y_test, predicciones_defecto)
    # *** CORRECCIÓN ***: Título del gráfico
    titulo = "Matriz de Confusión - SMOTE + RF (Umbral = 0.5)"
    _plot_confusion_matrix(matriz_confusion, titulo)

def paso_5_optimizar_umbral(mejor_modelo, X_train, y_train, n_splits, precision_minima, random_state):
    """
    Busca el umbral de decisión óptimo usando predicciones "Out-of-Fold" (OOF)
    Y ADEMÁS, grafica la relación entre el umbral, la precisión y el recall.
    """
    print(f"\nOptimizando umbral para: MAX(Recall) sujeto a Precision >= {precision_minima}...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print("Obteniendo predicciones de validación cruzada (OOF)...")
    y_probas_cv = cross_val_predict(
        mejor_modelo,
        X_train,
        y_train,
        cv=skf,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    # Listas para almacenar los valores para la gráfica
    lista_umbrales = np.linspace(0.01, 0.99, 1000)
    lista_precision = []
    lista_recall = []

    # Lógica para encontrar el mejor umbral (idéntica a la original)
    best_recall = -1
    optimal_threshold = 0.5 # Valor de fallback
    best_precision_at_best_recall = 0

    print("Calculando métricas para cada umbral...")
    for thresh in lista_umbrales:
        y_pred_thresh = np.where(y_probas_cv >= thresh, 1, 0)

        prec = precision_score(y_train, y_pred_thresh, zero_division=0)
        rec = recall_score(y_train, y_pred_thresh, zero_division=0)

        # Almacenar valores para la gráfica
        lista_precision.append(prec)
        lista_recall.append(rec)

        # Lógica de optimización
        if prec >= precision_minima:
            if rec > best_recall:
                best_recall = rec
                optimal_threshold = thresh
                best_precision_at_best_recall = prec
            elif rec == best_recall:
                # Si el recall es el mismo, preferimos el umbral más bajo (menos restrictivo)
                optimal_threshold = min(optimal_threshold, thresh)

    if best_recall == -1:
        print(f"¡ADVERTENCIA! No se encontró NINGÚN umbral que cumpla 'Precision >= {precision_minima}'.")
        print("El modelo no puede satisfacer este requisito. Usando 0.5 por defecto.")
        optimal_threshold = 0.5
    else:
        print("¡Éxito! Se encontró un umbral que cumple los requisitos.")
        print(f"   -> Umbral óptimo: {optimal_threshold:.4f}")
        print(f"   -> Recall resultante (en CV): {best_recall:.4f}")
        print(f"   -> Precision resultante (en CV): {best_precision_at_best_recall:.4f}")

    # --- INICIO DE LA NUEVA SECCIÓN DE GRÁFICO ---
    print("Generando gráfico Precision-Recall vs. Umbral...")
    plt.figure(figsize=(12, 7))
    
    # Graficar las curvas
    plt.plot(lista_umbrales, lista_precision, label='Precision', color='blue', linewidth=2)
    plt.plot(lista_umbrales, lista_recall, label='Recall', color='green', linewidth=2)
    
    # Línea vertical para el umbral óptimo encontrado
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Umbral Óptimo ({optimal_threshold:.4f})')
    
    # Línea horizontal para la restricción de precisión
    plt.axhline(y=precision_minima, color='gray', linestyle=':', 
                label=f'Precisión Mínima ({precision_minima})')
    
    # Configuración del gráfico
    plt.title('Precision y Recall vs. Umbral de Decisión (en datos de CV)', fontsize=16)
    plt.xlabel('Umbral de Decisión (Threshold)', fontsize=12)
    plt.ylabel('Puntuación (0.0 a 1.0)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    ticks = np.arange(0, 1, 0.05)
    # Aplica esas marcas a ambos ejes
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.ylim(0, 1.05) # Dejar un pequeño margen superior
    plt.xlim(0, 1.0)
    plt.show()
    # --- FIN DE LA NUEVA SECCIÓN DE GRÁFICO ---

    return optimal_threshold

def paso_6_evaluacion_final_y_guardado(mejor_modelo, X_test, y_test, scaler, optimal_threshold, feature_names):
    """
    Evaluación final en el Test set: Reporte, Matriz de Confusión,
    Curva ROC, Análisis de Errores y Guardado del modelo.
    """
    print("\n--- Evaluación Final en Conjunto de Prueba (Test Set) ---")
    
    predicciones_test_proba = mejor_modelo.predict_proba(X_test)[:, 1]
    predicciones_test_binarias = np.where(predicciones_test_proba >= optimal_threshold, 1, 0)

    # --- 1. Reporte de Clasificación ---
    print("\nReporte de Clasificación (Test Set):")
    target_names = ['0: Sin Defecto', '1: Con Defecto (Pegado)']
    print(classification_report(y_test, predicciones_test_binarias, target_names=target_names))

    # --- 2. Matriz de Confusión (Umbral Óptimo) ---
    matriz_confusion_opt = confusion_matrix(y_test, predicciones_test_binarias)
    titulo = f"Matriz de Confusión - Balanced RandomForest (Umbral Óptimo = {optimal_threshold:.4f})"
    _plot_confusion_matrix(matriz_confusion_opt, titulo)

    # --- 3. Curva ROC ---
    # 
    # *** CORRECCIÓN CRÍTICA ***: Usar probabilidades, no predicciones binarias.
    fpr, tpr, _ = metrics.roc_curve(y_test, predicciones_test_proba)
    auc_score = metrics.roc_auc_score(y_test, predicciones_test_proba)
    
    # --- SUAVIZADO VISUAL (Interpolación) ---
    # Agrupar por FPR y tomar el máximo TPR para evitar escalones verticales
    # y asegurar una función inyectiva para la interpolación.
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    df_roc_unique = df_roc.groupby('fpr')['tpr'].max().reset_index()
    fpr_clean = df_roc_unique['fpr'].values
    tpr_clean = df_roc_unique['tpr'].values

    # Generar puntos suaves e interpolar
    fpr_smooth = np.linspace(0, 1, 300)
    if len(fpr_clean) > 3:
        interpolator = PchipInterpolator(fpr_clean, tpr_clean)
        tpr_smooth = interpolator(fpr_smooth)
    else:
        tpr_smooth = np.interp(fpr_smooth, fpr_clean, tpr_clean)

    plt.figure()
    plt.plot(fpr_smooth, tpr_smooth, label=f"Balanced RandomForest (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Clasificador Aleatorio (AUC = 0.5)")
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC Suavizada (Test Set)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.4)
    plt.show()

    # ==========================================================================
    # PASO 4: Análisis de Errores (FN y FP)
    # ==========================================================================
    print("\n--- INICIANDO ANÁLISIS DE ERRORES EN EL TEST SET ---")

    # 1. Crear el DataFrame de análisis BASADO en y_test.
    # y_test SÍ conserva el índice original del DataFrame (ej. 145, 182, etc.)
    df_analisis = pd.DataFrame(y_test)

    # 2. Añadir las probabilidades y predicciones.
    # Como son arrays de numpy, se asignarán en orden a las filas existentes.
    df_analisis['Probabilidad_Defecto'] = predicciones_test_proba
    df_analisis['Prediccion_Binaria'] = predicciones_test_binarias
        
    #--- FALSOS NEGATIVOS (FN) ---
    # (Usamos 'Etiqueta_Defecto' como se llama en 'y_test')
    condicion_FN = (df_analisis['Etiqueta_Defecto'] == 1) & (df_analisis['Prediccion_Binaria'] == 0)
    falsos_negativos = df_analisis[condicion_FN]
 
    print(f"\n[INFORME] Se han encontrado {len(falsos_negativos)} Falsos Negativos (Defectos NO detectados):")
    # Imprimir solo las columnas relevantes
    print(falsos_negativos[['Etiqueta_Defecto', 'Prediccion_Binaria', 'Probabilidad_Defecto']].to_string())

    # --- FALSOS POSITIVOS (FP) ---
    condicion_FP = (df_analisis['Etiqueta_Defecto'] == 0) & (df_analisis['Prediccion_Binaria'] == 1)
    falsos_positivos = df_analisis[condicion_FP]
 
    print(f"\n[INFORME] Se han encontrado {len(falsos_positivos)} Falsos Positivos (Falsas Alarmas):")
    print(falsos_positivos[['Etiqueta_Defecto', 'Prediccion_Binaria', 'Probabilidad_Defecto']].to_string())

    # --- 5. Guardar Artefactos del Modelo ---
    print("\nGuardando pipeline COMPLETO (Scaler+Selector+Modelo) y umbral...")
    
    artefactos_modelo = {
        "pipeline_completo": mejor_modelo, # ¡Aquí va todo junto!
        "umbral": optimal_threshold,
        "feature_names_originales": feature_names # Guardamos los nombres para referencia futura
    }
    
    # Ajusta el nombre del archivo según el modelo que estés usando (RF o CatBoost)
    nombre_archivo = 'modelo_con_umbral_PEGADOS_PipelineCompleto.pkl'
    
    with open(nombre_archivo, 'wb') as f:
        pickle.dump(artefactos_modelo, f)

    print(f"¡Proceso completado! Modelo guardado con umbral = {optimal_threshold:.4f}")

def _plot_confusion_matrix(cm, title):
    """Función de ayuda interna para graficar una matriz de confusión."""
    fig, ax = plt.subplots()
    sns.heatmap(
        pd.DataFrame(cm),
        annot=True,
        cmap="YlGnBu",
        fmt='g',
        cbar=False,
        xticklabels=["Predicho Sin Defecto", "Predicho Pegado"],
        yticklabels=["Real Sin Defecto", "Real Pegado"]
    )
    ax.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.title(title)
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.show()
    
def paso_extra_graficar_bias_varianza(modelo, X, y, cv, scoring_metric):
    """
    Genera la curva de aprendizaje para visualizar Bias y Varianza.
    
    La curva de aprendizaje es fundamental para diagnosticar problemas del modelo:
    - Si TRAIN score es ALTO y VALIDATION score es BAJO → OVERFITTING (alta varianza)
      Solución: regularización, menos features, más datos
      
    - Si TRAIN score es BAJO y VALIDATION score es BAJO → UNDERFITTING (alto bias)
      Solución: modelo más complejo, más features, ajustar hiperparámetros
      
    - Si ambas curvas convergen a un buen score → modelo equilibrado
    
    Parámetros:
    -----------
    modelo : Pipeline entrenado
    X : DataFrame de características (usamos training set)
    y : Series de etiquetas (usamos training set)
    cv : Objeto StratifiedKFold para validación cruzada
    scoring_metric : Métrica a evaluar (F2 Score, etc.)
    """
    print("\nGenerando Curvas de Aprendizaje...")
    
    # Usar learning_curve de sklearn para entrenar con tamaños de datos progresivos
    # Esto permite ver cómo mejora el modelo a medida que aumentan los datos
    train_sizes, train_scores, val_scores = learning_curve(
        modelo, 
        X, 
        y, 
        cv=cv,                                    # Validación cruzada estratificada
        scoring=scoring_metric,                   # Métrica a medir (F2, recall, etc.)
        n_jobs=-1,                                # Usar todos los núcleos CPU
        train_sizes=np.linspace(0.1, 1.0, 10),   # 10 puntos: 10%, 20%, ..., 100%
        shuffle=True                              # Mezclar datos antes de entrenar
    )

    # Calcular estadísticas para cada tamaño de entrenamiento
    # train_scores tiene shape (10, 5) si usamos 5-fold CV
    # Promediamos los 5 scores para obtener un solo valor por tamaño
    train_mean = np.mean(train_scores, axis=1)    # Promedio de scores de entrenamiento
    train_std = np.std(train_scores, axis=1)      # Desviación estándar (variabilidad)
    val_mean = np.mean(val_scores, axis=1)        # Promedio de scores de validación
    val_std = np.std(val_scores, axis=1)          # Desviación estándar

    # Crear figura para el gráfico
    plt.figure(figsize=(10, 6))
    
    # --- CURVA DE ENTRENAMIENTO ---
    # Muestra cómo mejora el modelo con más datos de entrenamiento
    # Normalmente, aumentar datos → mejora el score (la curva sube)
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Score Entrenamiento")
    # Área sombreada alrededor de la línea = ±1 desviación estándar
    # Muestra la variabilidad entre los diferentes folds
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    
    # --- CURVA DE VALIDACIÓN ---
    # Muestra cómo generaliza el modelo a datos nuevos (no vistos en entrenamiento)
    # Esta es la métrica MÁS importante (desempeño real esperado)
    plt.plot(train_sizes, val_mean, 'o-', color="g", label="Score Validación (CV)")
    # Área sombreada = variabilidad de la validación entre folds
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")

    # Configuración del gráfico
    plt.title("Curva de Aprendizaje (Bias vs Varianza) - F2 Score")
    plt.xlabel("Tamaño del Set de Entrenamiento (muestras)")
    plt.ylabel("Score (0 a 1)")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    # ==== INTERPRETACIÓN AUTOMÁTICA DEL GAP TRAIN-VALIDACIÓN ====
    # El "gap" (brecha) entre el score de entrenamiento y validación es un indicador
    # clave del estado del modelo:
    # - Gap pequeño (< 0.05) → El modelo generaliza bien, sin overfitting
    # - Gap moderado (0.05-0.10) → Overfitting leve, aceptable en muchos casos
    # - Gap grande (> 0.10) → Overfitting significativo, necesita regularización
    
    # Calcular la brecha final (diferencia entre último punto de Train y Validación)
    # Un valor positivo significa que Train > Validación (típico en overfitting)
    gap_final = train_mean[-1] - val_mean[-1]
    
    # El sesgo aproximado se estima como (1 - mejor_score_validación)
    # Un sesgo alto significa que el modelo ni siquiera en entrenamiento logra buen score
    # Esto indicaría que el modelo es demasiado simple (underfitting)
    bias_final = 1 - val_mean[-1]
    
    # La varianza aproximada se asimila al gap train-validación
    # Una varianza alta significa que el modelo memoriza entrenamiento pero falla en validación
    # Esto indicaría que el modelo es demasiado complejo (overfitting)
    varianza_final = gap_final

    # --- IMPRIMIR MÉTRICAS DE DIAGNOSIS ---
    print(f"Gap final entre Train y Validación: {gap_final:.4f}")
    print(f"Sesgo (aprox): {bias_final:.4f}")
    print(f"Varianza (aprox): {varianza_final:.4f}")

    # --- DIAGNOSIS AUTOMÁTICA DEL ESTADO DEL MODELO ---
    # Basada en el tamaño del gap, proporcionar recomendaciones al usuario
    if gap_final < 0.05:
        # Caso IDEAL: El modelo generaliza bien
        # Train y Validación tienen scores similares → sin memorización
        print("✓ Gap pequeño: El modelo tiene buen balance (sin overfitting notable)")
    elif gap_final < 0.10:
        # Caso ACEPTABLE: Hay cierto overfitting pero dentro de límites tolerables
        # El modelo aprendió algunos patrones específicos del entrenamiento,
        # pero aún generaliza razonablemente bien a datos nuevos
        print("⚠ Gap moderado: Cierto overfitting, pero aceptable")
    else:
        # Caso PROBLEMÁTICO: Overfitting severo
        # El modelo funciona muy bien en entrenamiento pero falla en validación
        # Causas posibles: modelo muy complejo, pocos datos, sin regularización
        # Soluciones: aumentar regularización L2, reducir profundidad de árboles,
        # añadir más datos de entrenamiento, usar dropout
        print("✗ Gap grande: Overfitting significativo, considera regularización")


# ==============================================================================
# 5. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

def main():

    """
    Función principal que orquesta todo el pipeline de ML.
    """
    # PASO 1: Cargar y procesar los datos crudos
    X, y = paso_1_cargar_y_preparar_datos(FEATURE_NAMES)
    if X is None:
        return

    # --- NUEVO: DIAGNÓSTICO DE ENERGÍA ---
    # Esto te dirá si el problema es "demasiado fácil" físicamente
    graficar_distribucion_energia(X, y)
    # -------------------------------------
    
        # --- NUEVO: VISUALIZACIÓN GLOBAL ---
    # Ver distribución antes de decidir eliminar nada
    visualizar_datos_previos(X, y, FEATURE_NAMES)

    # PASO 2: Dividir y escalar los datos
    X_train, X_test, y_train, y_test = paso_2_escalar_y_dividir_datos(
        X, y, TEST_SIZE_RATIO, RANDOM_STATE_SEED
    )

    # PASO 3: Entrenar el pipeline SMOTE + RandomForest
    mejor_modelo = paso_3_entrenar_modelo(
        X_train, y_train, 
        N_SPLITS_CV, FBETA_BETA, RANDOM_STATE_SEED
    )

    # PASO 4: Evaluación inicial (Importancia, CM con umbral 0.5)
    paso_4_evaluar_importancia_y_umbral_defecto(
        mejor_modelo, X_test, y_test, FEATURE_NAMES
    )

    # PASO 5: Optimizar el umbral de decisión (con PRECISION_MINIMA)
    optimal_threshold = paso_5_optimizar_umbral(
        mejor_modelo, X_train, y_train, 
        N_SPLITS_CV, PRECISION_MINIMA, RANDOM_STATE_SEED
    )

    # PASO 6: Evaluación final (Reporte, CM, ROC, Errores) y Guardado
    paso_6_evaluacion_final_y_guardado(
        mejor_modelo, X_test, y_test, None, optimal_threshold, FEATURE_NAMES
    )
    
     # ==========================================================================
    # PASO 7 (BONUS): ANÁLISIS DE SESGO-VARIANZA (BIAS-VARIANCE TRADE-OFF)
    # ==========================================================================
    # Este paso NO es necesario para poner el modelo en producción,
    # pero es MUY ÚTIL para diagnosticar problemas de desempeño.
    #
    # ¿Qué es la curva de aprendizaje?
    # - Gráfico que muestra cómo mejora el modelo conforme aumentan los datos
    # - Eje X: Tamaño del set de entrenamiento (10%, 20%, ..., 100%)
    # - Eje Y: Score del modelo (F2-score en este caso)
    #
    # Interpretación:
    # - Si ambas curvas (Train y Validación) están ALTAS y JUNTAS:
    #   ✓ Buen modelo, buen balance entre sesgo y varianza
    #
    # - Si curva TRAIN está ALTA pero VALIDACIÓN está BAJA:
    #   ✗ OVERFITTING: El modelo memorizó el entrenamiento pero no generaliza
    #   Soluciones: Más regularización, menos features, más datos
    #
    # - Si ambas curvas están BAJAS:
    #   ✗ UNDERFITTING: El modelo es demasiado simple para el problema
    #   Soluciones: Modelo más complejo, más features, menos regularización
    print("\n" + "="*70)
    print("PASO 7 (BONUS): ANALIZANDO SESGO-VARIANZA (CURVA DE APRENDIZAJE)")
    print("="*70)
    print("Este gráfico muestra cómo mejora el modelo conforme aumentan los datos")
    
    # Preparar los objetos necesarios para la gráfica
    # CV: Usar la misma validación cruzada que en el Paso 3 (5 folds estratificados)
    cv_plot = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE_SEED)
    
    # Scorer: Usar la misma métrica de optimización (F2-score)
    # Esto asegura que la gráfica mida lo mismo que el modelo fue entrenado
    f2_scorer = make_scorer(fbeta_score, beta=FBETA_BETA)

    # Generar la gráfica de aprendizaje
    # Nota: Usamos X_train e y_train para ver la curva de aprendizaje en datos de entrenamiento
    # (es decir, cómo habría rendido con menos datos durante el entrenamiento)
    paso_extra_graficar_bias_varianza(
        modelo=mejor_modelo,       # Pipeline entrenado en Paso 3
        X=X_train,                 # Características de entrenamiento
        y=y_train,                 # Etiquetas de entrenamiento
        cv=cv_plot,                # Validación cruzada estratificada
        scoring_metric=f2_scorer   # F2-score (mismo que usó GridSearchCV)
    )


if __name__ == "__main__":
    main()