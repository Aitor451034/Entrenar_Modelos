"""
Script para la Verificación de Estabilidad del Modelo (Logistic Regression CV).

Este script ejecuta el pipeline de entrenamiento completo (Lasso L1 + Opt. Umbral)
múltiples veces con diferentes semillas aleatorias ('random_state')
para evaluar la estabilidad de las métricas del modelo (Recall, Precisión)
Y la estabilidad del propio umbral de decisión.

Un modelo estable producirá resultados similares independientemente
de cómo se dividan los datos.
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
import warnings

# --- Funciones científicas y estadísticas ---
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

# --- Componentes de Scikit-learn ---
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    fbeta_score, make_scorer, confusion_matrix,
    precision_score, recall_score
)

# Ignorar advertencias de convergencia
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# ==============================================================================
# 2. CONFIGURACIÓN DE LA PRUEBA DE ESTABILIDAD
# ==============================================================================

# --- Configuración del Bucle ---
N_ITERACIONES_ESTABILIDAD = 20  # Número de veces que se re-entrenará el modelo
TEST_SIZE_RATIO = 0.4           # Mismo Test Size que en tu script original
METRICA_FOCO = 'Recall'         # Métrica principal para el veredicto ('Recall' o 'Precision')

# --- Parámetros del Modelo (copiados de tu script) ---
N_SPLITS_CV_MODELO = 5          # Folds para el LogisticRegressionCV
PRECISION_MINIMA_OBJETIVO = 0.70 # Tu regla para el umbral

# --- Regla de Decisión (Veredicto) ---
UMBRAL_DESVIACION_ESTABLE = 0.05  # Desviación estándar máxima para considerar "ESTABLE"


# ==============================================================================
# 3. FUNCIONES DE CARGA Y EXTRACCIÓN DE CARACTERÍSTICAS
# ==============================================================================
# (Copiadas directamente de tu script de Regresión Logística)

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

def calcular_parametros(df):
    """Función principal de carga y "Feature Engineering"."""
    if df is None:
        print("No se pudo cargar el archivo. Terminando ejecución.")
        return np.array([]), np.array([])
    
    # PREPROCESAMIENTO INICIAL
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
            
    # BUCLE DE FEATURE ENGINEERING
    X_calculado, y_calculado = [], []
    print(f"Iniciando cálculo de features para {len(new_df)} puntos...")
    for i in new_df.index:
        # (Lógica de extracción de features copiada idénticamente)
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
        resistencia_ultima_val = np.nan_to_num(r_e, nan=0)
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
        desv_R_pre_max = np.nan_to_num(np.std(valores_resistencia[:I_R_max]), nan=0)
        pendientes = calcular_pendiente(valores_resistencia, t_soldadura)
        pendientes_post_max = pendientes[I_R_max:]
        pendientes_negativas_post = sum(1 for p in pendientes_post_max if p < 0)
        valores_resistencia_pre_max, valores_tiempo_pre_max = valores_resistencia[:I_R_max + 1], t_soldadura[:I_R_max + 1]
        area_pre_mitad = np.nan_to_num(np.trapz(valores_resistencia_pre_max, valores_tiempo_pre_max), nan=0)
        valores_resistencia_post_max, valores_tiempo_post_max = valores_resistencia[I_R_max:], t_soldadura[I_R_max:]
        area_post_mitad = np.nan_to_num(np.trapz(valores_resistencia_post_max, valores_tiempo_post_max), nan=0)
        try: desv_pre_mitad_t = np.nan_to_num(np.std(valores_resistencia_pre_max), nan=0)
        except ValueError: desv_pre_mitad_t = 0
        primera_derivada, segunda_derivada, tercera_derivada = calcular_derivadas(valores_resistencia, t_soldadura)
        try:
            max_curvatura = np.nan_to_num(np.max(np.abs(segunda_derivada)), nan=0)
            max_jerk = np.nan_to_num(np.max(np.abs(tercera_derivada)), nan=0)
            puntos_inflexion = np.where(np.diff(np.sign(segunda_derivada)))[0]
            num_puntos_inflexion = np.nan_to_num(len(puntos_inflexion), nan=0)
        except ValueError: max_curvatura, max_jerk, num_puntos_inflexion = 0, 0, 0
        try:
            mediana, varianza = np.nan_to_num(np.median(valores_resistencia), nan=0), np.nan_to_num(np.var(valores_resistencia), nan=0)
            rango_intercuartilico = np.nan_to_num(np.percentile(valores_resistencia, 75) - np.percentile(valores_resistencia, 25), nan=0)
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
            float(rango_intercuartilico), float(desv_pre_mitad_t), float(resistencia_ultima_val), float(desv), float(pendiente_V),
            float(rms), float(rango_rmax_rmin), float(r_mean_post_max), float(r_mean), float(desv_R_pre_max), float(pendientes_negativas_post),
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
    X_raw, y_raw = calcular_parametros(df_raw)
    
    if X_raw.size == 0:
        print("No se cargaron datos. Terminando.")
        return None, None
        
    X = pd.DataFrame(X_raw, columns=feature_names)
    y = pd.Series(y_raw, name="Etiqueta_Defecto")
    return X, y


# ==============================================================================
# 4. FUNCIÓN PRINCIPAL DE VERIFICACIÓN DE ESTABILIDAD
# ==============================================================================

def encontrar_umbral_optimo(modelo, X_train, y_train, cv_folds, precision_minima):
    """
    Función de ayuda para replicar el "PASO 6" de tu script:
    Optimización del umbral de decisión.
    """
    # 1. Obtener probabilidades "fuera de fold" (OOF)
    y_probas_cv = cross_val_predict(
        modelo, 
        X_train, 
        y_train, 
        cv=cv_folds, 
        method='predict_proba', 
        n_jobs=1 # n_jobs=-1 puede dar problemas dentro de un bucle, forzamos a 1
    )[:, 1]

    # 2. Iterar sobre posibles umbrales
    lista_umbrales = np.linspace(0.01, 0.99, 500) # 500 pasos es suficiente
    best_recall = -1
    optimal_threshold = 0.5 # Fallback
    
    for thresh in lista_umbrales:
        y_pred_thresh = np.where(y_probas_cv >= thresh, 1, 0)
        prec = precision_score(y_train, y_pred_thresh, zero_division=0)
        rec = recall_score(y_train, y_pred_thresh, zero_division=0)

        if prec >= precision_minima:
            if rec > best_recall:
                best_recall = rec
                optimal_threshold = thresh
            elif rec == best_recall:
                optimal_threshold = min(optimal_threshold, thresh)

    return optimal_threshold


def ejecutar_prueba_estabilidad(n_iteraciones, umbral_std_estable, metric_focus):
    """
    Ejecuta el pipeline de entrenamiento completo (Lasso L1 + Opt. Umbral)
    N veces con diferentes semillas aleatorias.
    """
    
    print("--- INICIANDO PRUEBA DE ESTABILIDAD DEL MODELO (LogisticRegressionCV) ---")
    
    # --- PASO 1: Cargar datos (UNA SOLA VEZ) ---
    X, y = cargar_datos_completos(RUTA_CSV_POR_DEFECTO, FEATURE_NAMES)
    if X is None:
        print("Error al cargar datos. Abortando prueba.")
        return

    lista_recalls = []
    lista_precisions = []
    lista_umbrales_optimos = []

    # --- PASO 2: Bucle de Verificación ---
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
        
        # (DataFrame para la búsqueda de umbral)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

        # 3. Entrenar el Modelo (PASO 4 de tu script)
        # 
        skf_cv = StratifiedKFold(n_splits=N_SPLITS_CV_MODELO, shuffle=True, random_state=seed)
        f2_scorer = make_scorer(fbeta_score, beta=2)
        lista_lambdas_log = np.logspace(-4, 1, 100) # Reducido a 100 para velocidad

        modelo_logreg = LogisticRegressionCV(
            Cs=lista_lambdas_log,
            cv=skf_cv,
            scoring=f2_scorer,
            penalty="l1",
            solver="liblinear",
            tol=1e-5,
            max_iter=3000,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1
        )
        modelo_logreg.fit(X_train_scaled, y_train)

        # 4. Optimizar Umbral (PASO 6 de tu script)
        # Usamos los datos de TRAIN para encontrar el umbral
        umbral_optimo_iter = encontrar_umbral_optimo(
            modelo_logreg, 
            X_train_scaled_df, # cross_val_predict funciona mejor con DataFrames
            y_train, 
            cv_folds=skf_cv, 
            precision_minima=PRECISION_MINIMA_OBJETIVO
        )
        lista_umbrales_optimos.append(umbral_optimo_iter)

        # 5. Evaluar en el Test Set (PASO 7 de tu script)
        predicciones_test_proba = modelo_logreg.predict_proba(X_test_scaled)[:, 1]
        predicciones_test_binarias = np.where(predicciones_test_proba >= umbral_optimo_iter, 1, 0)
        
        # Calcular métricas (solo para la clase 1, "Defecto")
        recall = recall_score(y_test, predicciones_test_binarias, pos_label=1, zero_division=0)
        precision = precision_score(y_test, predicciones_test_binarias, pos_label=1, zero_division=0)

        print(f"Resultado Iteración {i+1}: Umbral={umbral_optimo_iter:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
        lista_recalls.append(recall)
        lista_precisions.append(precision)

    # --- PASO 3: Analizar Resultados Agregados ---
    print("\n\n--- RESULTADOS FINALES DE ESTABILIDAD (LogisticRegressionCV) ---")
    
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

    # ¡NUEVO! Análisis del UMBRAL
    media_umbral = np.mean(lista_umbrales_optimos)
    std_umbral = np.std(lista_umbrales_optimos)
    min_umbral = np.min(lista_umbrales_optimos)
    max_umbral = np.max(lista_umbrales_optimos)

    print("\n--- Métricas del Umbral de Decisión Optimizado ---")
    print(f"Umbral Medio:           {media_umbral:.4f}  (Promedio)")
    print(f"Desviación Estándar:    {std_umbral:.4f}  (Estabilidad del umbral)")
    print(f"Rango de Umbrales:      {min_umbral:.4f} a {max_umbral:.4f}")


    # --- PASO 4: Veredicto Final ---
    print("\n--- VEREDICTO ---")
    
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
        print("El rendimiento del modelo es consistente y fiable.")
    else:
        print(f"[ VEREDICTO: INESTABLE ]")
        print(f"La desviación estándar de {metric_focus} ({std_foco:.4f}) es MAYOR que el umbral aceptable (> {umbral_std_estable}).")
        print("El rendimiento del modelo varía significativamente dependiendo de la división de los datos.")
        print("RECOMENDACIÓN: El modelo no es fiable para producción. Se necesita más datos.")

    # --- PASO 5: Visualización ---
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Distribución de Resultados (LogisticRegressionCV) tras {n_iteraciones} Iteraciones', fontsize=16)

    # Gráfico de Recall
    sns.histplot(lista_recalls, kde=True, ax=ax[0], bins=10, color='blue')
    ax[0].axvline(media_recall, color='red', linestyle='--', label=f'Media: {media_recall:.3f}')
    ax[0].axvline(min_recall, color='black', linestyle=':', label=f'Mín: {min_recall:.3f}')
    ax[0].set_title(f'Estabilidad del Recall (Clase 1)')
    ax[0].set_xlabel('Recall')
    ax[0].legend()

    # Gráfico de Precision
    sns.histplot(lista_precisions, kde=True, ax=ax[1], bins=10, color='green')
    ax[1].axvline(media_precision, color='red', linestyle='--', label=f'Media: {media_precision:.3f}')
    ax[1].axvline(min_precision, color='black', linestyle=':', label=f'Mín: {min_precision:.3f}')
    ax[1].set_title(f'Estabilidad de la Precision (Clase 1)')
    ax[1].set_xlabel('Precision')
    ax[1].legend()
    
    # Gráfico de Umbral
    sns.histplot(lista_umbrales_optimos, kde=True, ax=ax[2], bins=10, color='purple')
    ax[2].axvline(media_umbral, color='red', linestyle='--', label=f'Media: {media_umbral:.3f}')
    ax[2].set_title(f'Estabilidad del Umbral Optimizado')
    ax[2].set_xlabel('Umbral')
    ax[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==============================================================================
# 5. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    ejecutar_prueba_estabilidad(
        n_iteraciones=N_ITERACIONES_ESTABILIDAD,
        umbral_std_estable=UMBRAL_DESVIACION_ESTABLE,
        metric_focus=METRICA_FOCO
    )
    