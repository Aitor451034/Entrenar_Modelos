"""
Script para la Verificación de Estabilidad del Modelo (Bootstrapping).

Este script ejecuta el pipeline de entrenamiento (SMOTE + RandomForest)
múltiples veces con diferentes semillas aleatorias ('random_state')
para evaluar la estabilidad de las métricas del modelo (Recall y Precisión).

Un modelo estable producirá métricas similares independientemente
de cómo se dividan los datos de entrenamiento/prueba.
"""

# ==============================================================================
# 1. IMPORTACIONES DE BIBLIOTECAS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Ignorar advertencias de convergencia que pueden aparecer en algunas iteraciones
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


# ==============================================================================
# 2. CONFIGURACIÓN DE LA PRUEBA DE ESTABILIDAD
# ==============================================================================

# --- ¡¡IMPORTANTE!! ---
# Pega aquí el diccionario de 'Mejores parámetros encontrados'
# que obtuviste de tu script de GridSearchCV.
# Ejemplo: {'model__max_depth': 7, 'model__min_samples_leaf': 3, ...}
#
# (He puesto los valores de tu 'param_grid' como ejemplo, 
#  DEBES reemplazarlos por tus resultados reales)
#
MEJORES_PARAMETROS_GRID = {
    'model__max_depth': 5,
    'model__max_features': 'sqrt',
    'model__min_samples_leaf': 3,
    'model__min_samples_split': 2,
    'model__n_estimators': 200
}


# --- Configuración del Bucle ---
N_ITERACIONES_ESTABILIDAD = 20  # Número de veces que se re-entrenará el modelo (Recomendado: 10-30)
TEST_SIZE_RATIO = 0.4           # Mismo Test Size que en tu script original
METRICA_FOCO = 'Recall'         # Métrica principal para el veredicto (Puede ser 'Recall' o 'Precision')

# --- Regla de Decisión (Veredicto) ---
# ¿Qué desviación estándar (std) consideramos "estable"?
# 0.05 significa que la mayoría de los resultados (68%)
# caen en un rango de +/- 5% alrededor de la media.
# (ej. si la media es 0.80, los resultados suelen estar entre 0.75 y 0.85)
UMBRAL_DESVIACION_ESTABLE = 0.05 


# ==============================================================================
# 3. FUNCIONES DE CARGA Y EXTRACCIÓN DE CARACTERÍSTICAS
# ==============================================================================
# (Copiadas directamente de tu script anterior para que este sea autónomo)

RUTA_CSV_POR_DEFECTO = r"C:\Users\U5014554\Desktop\EntrenarModelo\DATA\Inputs_modelo_pegado_con_datos4_mas.csv"
FEATURE_NAMES = [
    "rango_r_beta_alfa", "rango_t_e_beta", "rango_r_e_beta", "resistencia_inicial", "k4", "k3",
    "rango_intercuartilico", "desv_pre_mitad_t", "resistencia_ultima", "desv", "pendiente_V",
    "rms", "rango_rmax_rmin", "r_mean_post_max", "r_mean", "desv_R_pre_max", "pendientes_negativas_post",
    "rango_tiempo_max_min", "area_bajo_curva", "area_pre_mitad", "area_post_mitad",
    "max_curvatura", "num_puntos_inflexion", "max_jerk", "mediana", "varianza", "asimetria",
    "curtosis", "num_picos", "num_valles", "q", "m_min_cuadrados"
]

def leer_archivo(ruta_csv_defecto):
    """Lee un archivo CSV con datos de soldadura."""
    print("Abriendo archivo ...")
    try:
        df = pd.read_csv(ruta_csv_defecto, encoding="utf-8", sep=";", on_bad_lines="skip", header=None, quotechar='"', decimal=",", skiprows=3)
        print("¡Archivo CSV leído correctamente!")
        return df
    except FileNotFoundError:
        print("No se ha encontrado el archivo en la ruta por defecto. Abriendo diálogo...")
        root = tk.Tk()
        root.withdraw()
        ruta_csv_manual = filedialog.askopenfilename(title="Seleccionar archivo que contiene los datos", filetypes=[("Archivos de CSV", "*.csv")])
        if not ruta_csv_manual: return None
        try:
            df = pd.read_csv(ruta_csv_manual, encoding="utf-8", sep=";", on_bad_lines="skip", header=None, quotechar='"', decimal=",")
            print("¡Archivo CSV leído correctamente!")
            return df
        except Exception as e:
            print(f"Se produjo un error al leer el archivo seleccionado: {e}")
            return None
    except Exception as e:
        print(f"Se produjo un error inesperado al leer el archivo: {e}")
        return None

def calcular_pendiente(resistencias, tiempos):
    """Calcula la pendiente (tasa de cambio) entre valores consecutivos."""
    if len(resistencias) <= 1 or len(tiempos) <= 1: return [0]
    pendientes = []
    for i in range(len(resistencias) - 1):
        delta_t = tiempos[i + 1] - tiempos[i]
        delta_r = resistencias[i + 1] - resistencias[i]
        if delta_t == 0: pendiente_actual = 0
        else: pendiente_actual = (delta_r / delta_t) * 100
        pendientes.append(round(np.nan_to_num(pendiente_actual, nan=0), 2))
    return pendientes

def calcular_derivadas(resistencias, tiempos):
    """Calcula la 1ra, 2da y 3ra derivada de la curva R(t)."""
    if len(resistencias) <= 1 or len(tiempos) <= 1: return np.array([0]), np.array([0]), np.array([0])
    primera_derivada = np.gradient(resistencias, tiempos)
    segunda_derivada = np.gradient(primera_derivada, tiempos)
    tercera_derivada = np.gradient(segunda_derivada, tiempos)
    return (np.nan_to_num(primera_derivada, nan=0), np.nan_to_num(segunda_derivada, nan=0), np.nan_to_num(tercera_derivada, nan=0))

def preprocesar_dataframe_inicial(df):
    """Limpia el DataFrame crudo."""
    new_df = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]]
    new_df = new_df.iloc[:-2]
    new_df.columns = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "KAI2", "Ts2", "Fuerza", "Etiqueta datos"]
    for col in ["KAI2", "Ts2", "Fuerza"]: new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    float_cols = new_df.select_dtypes(include='float64').columns
    new_df = new_df.round({col: 4 for col in float_cols})
    new_df.index = range(1, len(new_df) + 1)
    for col in df.columns:
        if df[col].dtype == object:
            try: df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except: pass
    return new_df

def extraer_features_fila_por_fila(new_df):
    """Itera sobre cada fila y calcula el vector de 32 características."""
    X_calculado, y_calculado = [], []
    print(f"Iniciando cálculo de features para {len(new_df)} puntos...")
    for i in new_df.index:
        datos_voltaje = new_df.loc[i, "Voltajes inst."]
        datos_corriente = new_df.loc[i, "Corrientes inst."]
        if pd.isna(datos_voltaje) or pd.isna(datos_corriente): continue
        valores_voltaje = [round(float(v), 0) for v in datos_voltaje.split(';') if v.strip()]
        valores_corriente = [round(0.001 if float(v) == 0 else float(v), 0) for v in datos_corriente.split(';') if v.strip()]
        valores_resistencia = [round(v / c if c != 0 else 0, 2) for v, c in zip(valores_voltaje, valores_corriente)]
        valores_resistencia.append(0)
        ns, ts2 = int(new_df.loc[i, "Ns"]), int(new_df.loc[i, "Ts2"])
        t_soldadura = (np.linspace(0, ts2, ns + 1)).tolist()
        if len(t_soldadura) != len(valores_resistencia):
            min_len = min(len(t_soldadura), len(valores_resistencia))
            t_soldadura, valores_resistencia = t_soldadura[:min_len], valores_resistencia[:min_len]
            valores_voltaje = valores_voltaje[:min_len]
        if not t_soldadura: continue
        resistencia_max, I_R_max = max(valores_resistencia), np.argmax(valores_resistencia)
        t_R_max = int(t_soldadura[I_R_max])
        r0, t0 = valores_resistencia[0], t_soldadura[0]
        r_e, t_e = valores_resistencia[-2], t_soldadura[-2]
        resistencia_min, t_min = min(valores_resistencia[:-1]), np.argmin(valores_resistencia[:-1])
        t_soldadura_min = t_soldadura[t_min]
        kAI2, f = new_df.loc[i, "KAI2"], new_df.loc[i, "Fuerza"]
        q = np.nan_to_num(((((kAI2 * 1000.0) ** 2) * (ts2 / 1000.0)) / (f * 10.0)), nan=0)
        area_bajo_curva = np.nan_to_num(np.trapz(valores_resistencia, t_soldadura), nan=0)
        resistencia_ultima = valores_resistencia[-2]
        try: delta_t_k4 = t_e - t_R_max; k4 = 0 if delta_t_k4 == 0 else ((r_e - resistencia_max) / delta_t_k4) * 100
        except ZeroDivisionError: k4 = 0
        k4 = np.nan_to_num(k4, nan=0)
        try: delta_t_k3 = t_R_max - t0; k3 = 0 if delta_t_k3 == 0 else ((resistencia_max - r0) / delta_t_k3) * 100
        except ZeroDivisionError: k3 = 0
        k3 = np.nan_to_num(k3, nan=0)
        desv = np.nan_to_num(np.std(valores_resistencia), nan=0)
        rms = np.nan_to_num(np.sqrt(np.mean(np.square(valores_resistencia))), nan=0)
        rango_tiempo_max_min = np.nan_to_num(t_R_max - t_soldadura_min, nan=0)
        rango_rmax_rmin = np.nan_to_num(resistencia_max - resistencia_min, nan=0)
        voltaje_max, t_max_v = max(valores_voltaje), np.argmax(valores_voltaje)
        t_voltaje_max = t_soldadura[t_max_v]
        voltaje_final, t_voltaje_final = valores_voltaje[-2], t_soldadura[-2]
        try: delta_t_v = t_voltaje_max - t_voltaje_final; pendiente_V = 0 if delta_t_v == 0 else ((voltaje_max - voltaje_final) / delta_t_v)
        except ZeroDivisionError: pendiente_V = 0
        pendiente_V = np.nan_to_num(pendiente_V, nan=0)
        r_mean_post_max = np.nan_to_num(np.mean(valores_resistencia[I_R_max:]), nan=0)
        resistencia_inicial = np.nan_to_num(r0, nan=2000)
        r_mean = np.nan_to_num(np.mean(valores_resistencia[:-1]), nan=0)
        rango_r_beta_alfa, rango_r_e_beta, rango_t_e_beta = np.nan_to_num(resistencia_max - r0, nan=0), np.nan_to_num(r_e - resistencia_max, nan=0), np.nan_to_num(t_e - t_R_max, nan=0)
        desv_R = np.nan_to_num(np.std(valores_resistencia[:I_R_max]), nan=0)
        pendientes = calcular_pendiente(valores_resistencia, t_soldadura)
        pendientes_post_max = pendientes[I_R_max:]
        pendientes_negativas_post = sum(1 for p in pendientes_post_max if p < 0)
        valores_resistencia_hasta_R_max, valores_tiempo_hasta_R_max = valores_resistencia[:I_R_max + 1], t_soldadura[:I_R_max + 1]
        area_pre_mitad = np.nan_to_num(np.trapz(valores_resistencia_hasta_R_max, valores_tiempo_hasta_R_max), nan=0)
        valores_resistencia_desde_R_max, valores_tiempo_desde_R_max = valores_resistencia[I_R_max:], t_soldadura[I_R_max:]
        area_post_mitad = np.nan_to_num(np.trapz(valores_resistencia_desde_R_max, valores_tiempo_desde_R_max), nan=0)
        try: desv_pre_mitad_t = np.nan_to_num(np.std(valores_resistencia_hasta_R_max), nan=0)
        except ValueError: desv_pre_mitad_t = 0
        primera_derivada, segunda_derivada, tercera_derivada = calcular_derivadas(valores_resistencia, t_soldadura)
        try:
            max_curvatura = np.nan_to_num(np.max(np.abs(segunda_derivada)), nan=0)
            puntos_inflexion = np.where(np.diff(np.sign(segunda_derivada)))[0]
            num_puntos_inflexion = np.nan_to_num(len(puntos_inflexion), nan=0)
            max_jerk = np.nan_to_num(np.max(np.abs(tercera_derivada)), nan=0)
        except ValueError: max_curvatura, num_puntos_inflexion, max_jerk = 0, 0, 0
        try:
            mediana, varianza = np.nan_to_num(np.median(valores_resistencia), nan=0), np.nan_to_num(np.var(valores_resistencia), nan=0)
            rango_intercuartilico = np.nan_to_num((np.percentile(valores_resistencia, 75) - np.percentile(valores_resistencia, 25)), nan=0)
            asimetria, curtosis = np.nan_to_num(skew(valores_resistencia), nan=0), np.nan_to_num(kurtosis(valores_resistencia), nan=0)
        except (ValueError, IndexError): mediana, varianza, rango_intercuartilico, asimetria, curtosis = 0, 0, 0, 0, 0
        valores_resistencia_np = np.array(valores_resistencia)
        picos, _ = find_peaks(valores_resistencia_np, height=0)
        valles, _ = find_peaks(-valores_resistencia_np)
        num_picos, num_valles = np.nan_to_num(len(picos), nan=0), np.nan_to_num(len(valles), nan=0)
        t_mean, r_mean_ols = np.nan_to_num(np.mean(t_soldadura), nan=0), np.nan_to_num(np.mean(valores_resistencia), nan=0)
        numerador = sum((r_mean_ols - ri) * (t_mean - ti) for ri, ti in zip(valores_resistencia, t_soldadura))
        denominador = sum((r_mean_ols - ri) ** 2 for ri in valores_resistencia)
        m_min_cuadrados = 0 if denominador == 0 else (numerador / denominador)
        X_calculado.append([
            float(rango_r_beta_alfa), float(rango_t_e_beta), float(rango_r_e_beta), float(resistencia_inicial), float(k4), float(k3),
            float(rango_intercuartilico), float(desv_pre_mitad_t), float(resistencia_ultima), float(desv), float(pendiente_V),
            float(rms), float(rango_rmax_rmin), float(r_mean_post_max), float(r_mean), float(desv_R), float(pendientes_negativas_post),
            float(rango_tiempo_max_min), float(area_bajo_curva), float(area_pre_mitad), float(area_post_mitad),
            float(max_curvatura), float(num_puntos_inflexion), float(max_jerk), float(mediana), float(varianza), float(asimetria),
            float(curtosis), float(num_picos), float(num_valles), float(q), float(m_min_cuadrados)
        ])
        y_calculado.append(int(new_df.loc[i, "Etiqueta datos"]))
    print("Cálculo de features completado.")
    return np.array(X_calculado), np.array(y_calculado)

def cargar_datos_completos(ruta_csv_defecto, feature_names):
    """Carga y prepara los datos X e y una sola vez."""
    df_raw = leer_archivo(ruta_csv_defecto)
    if df_raw is None:
        return None, None
    df_preprocesado = preprocesar_dataframe_inicial(df_raw)
    X_raw, y_raw = extraer_features_fila_por_fila(df_preprocesado)
    
    if X_raw.size == 0:
        print("No se cargaron datos. Terminando.")
        return None, None
        
    X = pd.DataFrame(X_raw, columns=feature_names)
    y = pd.Series(y_raw, name="Etiqueta_Defecto")
    return X, y


# ==============================================================================
# 4. FUNCIÓN PRINCIPAL DE VERIFICACIÓN DE ESTABILIDAD
# ==============================================================================

def ejecutar_prueba_estabilidad(n_iteraciones, umbral_std_estable, metric_focus):
    """
    Ejecuta el pipeline de entrenamiento completo N veces con diferentes
    semillas aleatorias y reporta la estabilidad de las métricas.
    """
    
    print("--- INICIANDO PRUEBA DE ESTABILIDAD DEL MODELO ---")
    
    # --- PASO 1: Cargar datos (UNA SOLA VEZ) ---
    X, y = cargar_datos_completos(RUTA_CSV_POR_DEFECTO, FEATURE_NAMES)
    if X is None:
        print("Error al cargar datos. Abortando prueba.")
        return

    # --- PASO 2: Preparar Parámetros del Modelo ---
    # Limpiamos los prefijos 'model__' para poder pasarlos
    # directamente al constructor de RandomForestClassifier.
    try:
        params_limpios = {
            key.replace('model__', ''): value 
            for key, value in MEJORES_PARAMETROS_GRID.items()
        }
    except Exception as e:
        print(f"Error: El diccionario 'MEJORES_PARAMETROS_GRID' no es válido: {e}")
        print("Por favor, pégalo exactamente como lo da GridSearchCV.")
        return
        
    print(f"Iniciando {n_iteraciones} iteraciones con estos parámetros:")
    print(params_limpios)

    lista_recalls = []
    lista_precisions = []

    # --- PASO 3: Bucle de Verificación ---
    for i in range(n_iteraciones):
        # Usamos 'i' como la semilla aleatoria para esta iteración
        seed = i
        print(f"\n--- Iteración {i+1}/{n_iteraciones} (Seed={seed}) ---")

        # 1. Dividir los datos (diferente cada vez)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE_RATIO, 
            random_state=seed, # <-- Semilla variable
            stratify=y
        )

        # 2. Escalar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. Definir el Pipeline (con la semilla variable)
        # 
        pipeline_final = ImbPipeline([
            ('smote', SMOTE(random_state=seed)), # <-- Semilla variable
            ('model', RandomForestClassifier(random_state=seed, **params_limpios)) # <-- Semilla variable
        ])

        # 4. Entrenar el modelo
        pipeline_final.fit(X_train_scaled, y_train)

        # 5. Evaluar en el Test Set
        y_pred = pipeline_final.predict(X_test_scaled)
        
        # Calcular métricas (solo para la clase 1, "Defecto")
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

        print(f"Resultado Iteración {i+1}: Recall={recall:.4f}, Precision={precision:.4f}")
        lista_recalls.append(recall)
        lista_precisions.append(precision)

    # --- PASO 4: Analizar Resultados Agregados ---
    print("\n\n--- RESULTADOS FINALES DE ESTABILIDAD ---")
    
    # Análisis de RECALL
    media_recall = np.mean(lista_recalls)
    std_recall = np.std(lista_recalls)
    min_recall = np.min(lista_recalls)
    max_recall = np.max(lista_recalls)

    print("\n--- Métricas de Recall (Clase 1: Defecto) ---")
    print(f"Iteraciones Totales:    {n_iteraciones}")
    print(f"Recall Medio:           {media_recall:.4f}  (Promedio)")
    print(f"Desviación Estándar:    {std_recall:.4f}  (¡Clave!)")
    print(f"Peor Caso (Min Recall): {min_recall:.4f}")
    print(f"Mejor Caso (Max Recall): {max_recall:.4f}")

    # Análisis de PRECISION
    media_precision = np.mean(lista_precisions)
    std_precision = np.std(lista_precisions)
    min_precision = np.min(lista_precisions)
    max_precision = np.max(lista_precisions)

    print("\n--- Métricas de Precision (Clase 1: Defecto) ---")
    print(f"Precision Media:        {media_precision:.4f}  (Promedio)")
    print(f"Desviación Estándar:    {std_precision:.4f}")
    print(f"Peor Caso (Min Prec):   {min_precision:.4f}")
    print(f"Mejor Caso (Max Prec):  {max_precision:.4f}")

    # --- PASO 5: Veredicto Final ---
    print("\n--- VEREDICTO ---")
    
    # Seleccionar la métrica para el veredicto
    if metric_focus.lower() == 'recall':
        std_foco = std_recall
    elif metric_focus.lower() == 'precision':
        std_foco = std_precision
    else:
        print(f"Advertencia: 'METRICA_FOCO' no reconocida. Usando Recall.")
        std_foco = std_recall

    if std_foco <= umbral_std_estable:
        print(f"[ VEREDICTO: ESTABLE ]")
        print(f"La desviación estándar de {metric_focus} ({std_foco:.4f}) está DENTRO del umbral aceptable (<= {umbral_std_estable}).")
        print("El rendimiento del modelo es consistente y fiable a través de diferentes divisiones de datos.")
    else:
        print(f"[ VEREDICTO: INESTABLE ]")
        print(f"La desviación estándar de {metric_focus} ({std_foco:.4f}) es MAYOR que el umbral aceptable (> {umbral_std_estable}).")
        print("El rendimiento del modelo varía significativamente dependiendo de los datos de entrenamiento.")
        print("RECOMENDACIÓN: El modelo no es fiable para producción. Se necesita más datos (especialmente de defectos).")

    # --- PASO 6: Visualización ---
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Distribución de Resultados tras {n_iteraciones} Iteraciones', fontsize=16)

    # Gráfico de Recall
    sns.histplot(lista_recalls, kde=True, ax=ax[0], bins=10, color='blue')
    ax[0].axvline(media_recall, color='red', linestyle='--', label=f'Media: {media_recall:.3f}')
    ax[0].axvline(min_recall, color='black', linestyle=':', label=f'Mín: {min_recall:.3f}')
    ax[0].set_title(f'Estabilidad del Recall (Clase 1)')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Frecuencia')
    ax[0].legend()

    # Gráfico de Precision
    sns.histplot(lista_precisions, kde=True, ax=ax[1], bins=10, color='green')
    ax[1].axvline(media_precision, color='red', linestyle='--', label=f'Media: {media_precision:.3f}')
    ax[1].axvline(min_precision, color='black', linestyle=':', label=f'Mín: {min_precision:.3f}')
    ax[1].set_title(f'Estabilidad de la Precision (Clase 1)')
    ax[1].set_xlabel('Precision')
    ax[1].set_ylabel('Frecuencia')
    ax[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==============================================================================
# 5. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    if MEJORES_PARAMETROS_GRID.get('model__max_depth') == 7 and MEJORES_PARAMETROS_GRID.get('model__n_estimators') == 300:
        print("************************************************************")
        print("¡ADVERTENCIA! Estás usando los parámetros de EJEMPLO.")
        print("Por favor, edita el script y pega tus 'MEJORES_PARAMETROS_GRID'")
        print("reales que obtuviste de tu GridSearchCV.")
        print("************************************************************\n")

    ejecutar_prueba_estabilidad(
        n_iteraciones=N_ITERACIONES_ESTABILIDAD,
        umbral_std_estable=UMBRAL_DESVIACION_ESTABLE,
        metric_focus=METRICA_FOCO
    )