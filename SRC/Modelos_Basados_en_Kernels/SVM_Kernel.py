"""
Script de Detección de Defectos en Soldadura (RSW)
Arquitectura: Procesos Gaussianos de Clasificación (GPC) con ARD
Basado en: Informe de Investigación Estratégica para Regímenes de Datos Escasos (Small Data).

Cambios principales respecto a versión anterior:
- Eliminación de SMOTE (prevención de distorsión de manifold).
- Eliminación de RandomForest/XGBoost.
- Inclusión de StandardScaler dentro del bucle de validación.
- Uso de Kernel Matern Anisotrópico (ARD) + WhiteKernel.
- Optimización orientada a Probabilidades Calibradas.
"""

# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tkinter as tk
from tkinter import filedialog
import time

# --- Funciones científicas y estadísticas ---
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

# --- Scikit-learn Core ---
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_predict, RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    fbeta_score, make_scorer, classification_report, confusion_matrix,
    precision_score, recall_score, roc_curve, roc_auc_score
)

# --- GPC y Kernels ---
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance # Reemplazo para ARD
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# 2. CONFIGURACIÓN
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

# Configuración basada en el reporte
TEST_SIZE_RATIO = 0.30  # Reservamos un poco más para test dado N pequeño, pero balanceado
RANDOM_STATE_SEED = 42
N_SPLITS_CV = 5
N_REPEATS_CV = 5        # Robustez extra para Small Data
FBETA_BETA = 2          # Prioridad Recall
PRECISION_MINIMA = 0.70 # Restricción operativa estricta

# ==============================================================================
# 3. FUNCIONES DE PREPROCESAMIENTO (INVARIANTES)
# ==============================================================================
# Nota: Se mantienen las funciones de extracción de características originales.
# Solo se incluye el código necesario para que el script sea ejecutable.

def leer_archivo():
    print("Selecciona el archivo CSV que contiene los datos...")
    root = tk.Tk()
    root.withdraw()
    ruta_csv = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
    if not ruta_csv: return None
    try:
        # Ajustar skiprows según tu CSV real
        df = pd.read_csv(ruta_csv, encoding="utf-8", sep=";", on_bad_lines="skip", 
                         header=None, decimal=",", skiprows=3)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def calcular_pendiente(resistencias, tiempos):
    """Calcula la pendiente (tasa de cambio) entre valores consecutivos."""
    if len(resistencias) <= 1 or len(tiempos) <= 1:
        return [0]
    pendientes = []
    for i in range(len(resistencias) - 1):
        delta_t = tiempos[i + 1] - tiempos[i]
        delta_r = resistencias[i + 1] - resistencias[i]
        if delta_t == 0:
            pendiente_actual = 0
        else:
            pendiente_actual = (delta_r / delta_t) * 100
        pendientes.append(round(np.nan_to_num(pendiente_actual, nan=0), 2))
    return pendientes

def calcular_derivadas(resistencias, tiempos):
    """Calcula la 1ra, 2da y 3ra derivada de la curva R(t)."""
    if len(resistencias) <= 1 or len(tiempos) <= 1:
        return np.array([0]), np.array([0]), np.array([0])
    primera_derivada = np.gradient(resistencias, tiempos)
    segunda_derivada = np.gradient(primera_derivada, tiempos)
    tercera_derivada = np.gradient(segunda_derivada, tiempos)
    return (
        np.nan_to_num(primera_derivada, nan=0),
        np.nan_to_num(segunda_derivada, nan=0),
        np.nan_to_num(tercera_derivada, nan=0)
    )

def preprocesar_dataframe_inicial(df):
    """Limpia el DataFrame crudo."""
    new_df = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]]
    new_df = new_df.iloc[:-2]
    new_df.columns = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "KAI2", "Ts2", "Fuerza", "Etiqueta datos"]
    for col in ["KAI2", "Ts2", "Fuerza"]:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    float_cols = new_df.select_dtypes(include='float64').columns
    new_df = new_df.round({col: 4 for col in float_cols})
    new_df.index = range(1, len(new_df) + 1)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except:
                pass
    return new_df

def extraer_features_fila_por_fila(new_df):
    """Itera sobre cada fila y calcula el vector de 32 características."""
    X_calculado = []
    y_calculado = []
    print(f"Iniciando cálculo de features para {len(new_df)} puntos de soldadura...")

    for i in new_df.index:
        # --- 1. Leer series temporales ---
        datos_voltaje = new_df.loc[i, "Voltajes inst."]
        datos_corriente = new_df.loc[i, "Corrientes inst."]
        if pd.isna(datos_voltaje) or pd.isna(datos_corriente):
            print(f"Advertencia: Datos nulos en fila {i}. Saltando.")
            continue
            
        valores_voltaje = [float(v) for v in datos_voltaje.split(';') if v.strip()]
        valores_voltaje = [round(v, 0) for v in valores_voltaje]
        valores_corriente = [0.001 if float(v) == 0 else float(v) for v in datos_corriente.split(';') if v.strip()]
        valores_corriente = [round(v, 0) for v in valores_corriente]

        # --- 2. Calcular Resistencia y Tiempo ---
        valores_resistencia = [v / c if c != 0 else 0 for v, c in zip(valores_voltaje, valores_corriente)]
        valores_resistencia = [round(r, 2) for r in valores_resistencia]
        valores_resistencia.append(0)
        ns = int(new_df.loc[i, "Ns"])
        ts2 = int(new_df.loc[i, "Ts2"])
        t_soldadura = (np.linspace(0, ts2, ns + 1)).tolist()

        if len(t_soldadura) != len(valores_resistencia):
            min_len = min(len(t_soldadura), len(valores_resistencia))
            t_soldadura = t_soldadura[:min_len]
            valores_resistencia = valores_resistencia[:min_len]
            valores_voltaje = valores_voltaje[:min_len]
        if not t_soldadura:
            print(f"Advertencia: Fila {i} sin datos de series temporales. Saltando.")
            continue

        # --- 3. Puntos clave R(t) ---
        resistencia_max = max(valores_resistencia)
        I_R_max = np.argmax(valores_resistencia)
        t_R_max = int(t_soldadura[I_R_max])
        r0 = valores_resistencia[0]
        t0 = t_soldadura[0]
        r_e = valores_resistencia[-2]
        t_e = t_soldadura[-2]
        resistencia_min = min(valores_resistencia[:-1])
        t_min = np.argmin(valores_resistencia[:-1])
        t_soldadura_min = t_soldadura[t_min]
        
        # --- 4. Parámetros escalares ---
        kAI2 = new_df.loc[i, "KAI2"]
        f = new_df.loc[i, "Fuerza"]
        
        # --- 5. Cálculo de Features (Lógica original) ---
        q = np.nan_to_num(((((kAI2 * 1000.0) ** 2) * (ts2 / 1000.0)) / (f * 10.0)), nan=0)
        area_bajo_curva = np.nan_to_num(np.trapz(valores_resistencia, t_soldadura), nan=0)
        resistencia_ultima = valores_resistencia[-2]
        try:
            delta_t_k4 = t_e - t_R_max
            k4 = 0 if delta_t_k4 == 0 else ((r_e - resistencia_max) / delta_t_k4) * 100
        except ZeroDivisionError: k4 = 0
        k4 = np.nan_to_num(k4, nan=0)
        try:
            delta_t_k3 = t_R_max - t0
            k3 = 0 if delta_t_k3 == 0 else ((resistencia_max - r0) / delta_t_k3) * 100
        except ZeroDivisionError: k3 = 0
        k3 = np.nan_to_num(k3, nan=0)
        desv = np.nan_to_num(np.std(valores_resistencia), nan=0)
        rms = np.nan_to_num(np.sqrt(np.mean(np.square(valores_resistencia))), nan=0)
        rango_tiempo_max_min = np.nan_to_num(t_R_max - t_soldadura_min, nan=0)
        rango_rmax_rmin = np.nan_to_num(resistencia_max - resistencia_min, nan=0)
        voltaje_max = max(valores_voltaje)
        t_max_v = np.argmax(valores_voltaje)
        t_voltaje_max = t_soldadura[t_max_v]
        voltaje_final = valores_voltaje[-2]
        t_voltaje_final = t_soldadura[-2]
        try:
            delta_t_v = t_voltaje_max - t_voltaje_final
            pendiente_V = 0 if delta_t_v == 0 else ((voltaje_max - voltaje_final) / delta_t_v)
        except ZeroDivisionError: pendiente_V = 0
        pendiente_V = np.nan_to_num(pendiente_V, nan=0)
        r_mean_post_max = np.nan_to_num(np.mean(valores_resistencia[I_R_max:]), nan=0)
        resistencia_inicial = np.nan_to_num(r0, nan=2000)
        r_mean = np.nan_to_num(np.mean(valores_resistencia[:-1]), nan=0)
        rango_r_beta_alfa = np.nan_to_num(resistencia_max - r0, nan=0)
        rango_r_e_beta = np.nan_to_num(r_e - resistencia_max, nan=0)
        rango_t_e_beta = np.nan_to_num(t_e - t_R_max, nan=0)
        desv_R = np.nan_to_num(np.std(valores_resistencia[:I_R_max]), nan=0)
        pendientes = calcular_pendiente(valores_resistencia, t_soldadura)
        pendientes_post_max = pendientes[I_R_max:]
        pendientes_negativas_post = sum(1 for p in pendientes_post_max if p < 0)
        valores_resistencia_hasta_R_max = valores_resistencia[:I_R_max + 1]
        valores_tiempo_hasta_R_max = t_soldadura[:I_R_max + 1]
        area_pre_mitad = np.nan_to_num(np.trapz(valores_resistencia_hasta_R_max, valores_tiempo_hasta_R_max), nan=0)
        valores_resistencia_desde_R_max = valores_resistencia[I_R_max:]
        valores_tiempo_desde_R_max = t_soldadura[I_R_max:]
        area_post_mitad = np.nan_to_num(np.trapz(valores_resistencia_desde_R_max, valores_tiempo_desde_R_max), nan=0)
        try:
            desv_pre_mitad_t = np.nan_to_num(np.std(valores_resistencia_hasta_R_max), nan=0)
        except ValueError: desv_pre_mitad_t = 0
        primera_derivada, segunda_derivada, tercera_derivada = calcular_derivadas(valores_resistencia, t_soldadura)
        try:
            max_curvatura = np.nan_to_num(np.max(np.abs(segunda_derivada)), nan=0)
            puntos_inflexion = np.where(np.diff(np.sign(segunda_derivada)))[0]
            num_puntos_inflexion = np.nan_to_num(len(puntos_inflexion), nan=0)
            max_jerk = np.nan_to_num(np.max(np.abs(tercera_derivada)), nan=0)
        except ValueError: max_curvatura, num_puntos_inflexion, max_jerk = 0, 0, 0
        try:
            mediana = np.nan_to_num(np.median(valores_resistencia), nan=0)
            varianza = np.nan_to_num(np.var(valores_resistencia), nan=0)
            rango_intercuartilico = np.nan_to_num((np.percentile(valores_resistencia, 75) - np.percentile(valores_resistencia, 25)), nan=0)
            asimetria = np.nan_to_num(skew(valores_resistencia), nan=0)
            curtosis = np.nan_to_num(kurtosis(valores_resistencia), nan=0)
        except (ValueError, IndexError): mediana, varianza, rango_intercuartilico, asimetria, curtosis = 0, 0, 0, 0, 0
        valores_resistencia_np = np.array(valores_resistencia)
        picos, _ = find_peaks(valores_resistencia_np, height=0)
        valles, _ = find_peaks(-valores_resistencia_np)
        num_picos = np.nan_to_num(len(picos), nan=0)
        num_valles = np.nan_to_num(len(valles), nan=0)
        t_mean = np.nan_to_num(np.mean(t_soldadura), nan=0)
        r_mean_ols = np.nan_to_num(np.mean(valores_resistencia), nan=0)
        numerador = sum((r_mean_ols - ri) * (t_mean - ti) for ri, ti in zip(valores_resistencia, t_soldadura))
        denominador = sum((r_mean_ols - ri) ** 2 for ri in valores_resistencia)
        m_min_cuadrados = 0 if denominador == 0 else (numerador / denominador)

        # --- 6. Ensamblar vector de características y etiqueta ---
        X_calculado.append([
            float(rango_r_beta_alfa), float(rango_t_e_beta), float(rango_r_e_beta),
            float(resistencia_inicial), float(k4), float(k3),
            float(rango_intercuartilico), float(desv_pre_mitad_t), float(resistencia_ultima),
            float(desv), float(pendiente_V), float(rms),
            float(rango_rmax_rmin), float(r_mean_post_max), float(r_mean),
            float(desv_R), float(pendientes_negativas_post), float(rango_tiempo_max_min),
            float(area_bajo_curva), float(area_pre_mitad), float(area_post_mitad),
            float(max_curvatura), float(num_puntos_inflexion), float(max_jerk),
            float(mediana), float(varianza), float(asimetria),
            float(curtosis), float(num_picos), float(num_valles),
            float(q), float(m_min_cuadrados)
        ])
        y_calculado.append(int(new_df.loc[i, "Etiqueta datos"]))
        
    print("Cálculo de features completado.")
    return np.array(X_calculado), np.array(y_calculado)

# ==============================================================================
# 4. PIPELINE DE APRENDIZAJE BAYESIANO
# ==============================================================================

def paso_1_cargar_y_preparar_datos(feature_names):
    df_raw = leer_archivo()
    if df_raw is None: return None, None
    
    # Asumimos que tienes las funciones de extracción definidas arriba o importadas
    # Aquí simplifico llamando a las funciones que definiste en tu prompt original
    # IMPORTANTE: Asegúrate de que preprocesar_dataframe_inicial y extraer... estén disponibles
    try:
        df_pre = preprocesar_dataframe_inicial(df_raw)
        X_arr, y_arr = extraer_features_fila_por_fila(df_pre)
    except NameError:
        print("Error: Funciones de extracción no definidas en este contexto.")
        return None, None

    if X_arr.size == 0: return None, None
    
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="Etiqueta_Defecto")
    
    print(f"Datos cargados: {len(X)} muestras. Tasa defecto: {y.mean():.2%}")
    return X, y

def paso_2_dividir_datos(X, y):
    """División estratificada simple. El escalado se mueve al Pipeline."""
    return train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, 
        random_state=RANDOM_STATE_SEED, stratify=y
    )

def paso_3_entrenar_svm(X_train, y_train):
    """
    Entrena un SVM con Kernel RBF usando GridSearchCV para optimizar C y Gamma.
    """
    print(f"\n--- Entrenando SVM (Kernel RBF) con {X_train.shape[1]} variables ---")
    
    # 1. Pipeline: Escalado + SVM
    # probability=True es CRÍTICO para que luego funcione el paso 5 (Calibración)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
    ])

    # 2. Grid de Hiperparámetros
    # C: Penalización (C alto = ajusta mucho / C bajo = más suave)
    # gamma: Alcance (gamma alto = solo puntos cercanos influyen / bajo = alcance lejano)
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 0.1, 0.01]
    }

    # Validación Cruzada
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    f2_scorer = make_scorer(fbeta_score, beta=2) # Prioridad Recall

    search = GridSearchCV(
        pipeline, param_grid, cv=rskf, scoring=f2_scorer, n_jobs=-1, verbose=1
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"Mejor F2: {search.best_score_:.4f} | Tiempo: {elapsed_time:.2f}s")
    print(f"Mejores params: {search.best_params_}")

    return search.best_estimator_

def paso_4_analisis_relevancia_ard(modelo, feature_names):
    print("\n--- Análisis de Importancia (Permutation Importance) ---")
    
    # Calculamos qué tanto afecta al modelo "romper" cada variable
    result = permutation_importance(
        modelo, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1, scoring='recall'
    )
    
    sorted_idx = result.importances_mean.argsort()

    # Visualización
    plt.figure(figsize=(10, 8))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx]
    )
    plt.title("Importancia de Variables (SVM - Permutación)")
    plt.tight_layout()
    plt.show()
    
    # Retornamos los índices de las mejores variables para la selección posterior
    # (Invertimos porque argsort es ascendente)
    return sorted_idx[::-1]

def paso_5_optimizar_umbral_probabilistico(modelo, X_train, y_train):
    """
    MODIFICACIÓN: Optimización basada en probabilidades puras de GPC.
    Objetivo: Max(Recall) sujeto a Precision >= 0.7.
    """
    print(f"\n--- Calibración de Umbral (Restricción: Precisión >= {PRECISION_MINIMA}) ---")
    
    # CAMBIO CRÍTICO AQUÍ:
    # Usamos StratifiedKFold estándar (sin repeticiones) para que cross_val_predict funcione.
    # Esto garantiza que cada muestra tenga exactamente UNA predicción.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE_SEED)
    
    y_probas = cross_val_predict(
        modelo, X_train, y_train, cv=skf, method='predict_proba', n_jobs=-1
    )[:, 1]

    umbrales = np.linspace(0.1, 0.95, 1000)
    best_thresh = 0.5
    best_recall = 0.0
    final_precision = 0.0
    
    precision_list, recall_list = [], []

    found_feasible = False
    
    for t in umbrales:
        preds = (y_probas >= t).astype(int)
        p = precision_score(y_train, preds, zero_division=0)
        r = recall_score(y_train, preds, zero_division=0)
        
        precision_list.append(p)
        recall_list.append(r)
        
        if p >= PRECISION_MINIMA:
            found_feasible = True
            if r > best_recall:
                best_recall = r
                best_thresh = t
                final_precision = p
            # Si el recall es igual, preferimos umbral menor (más robusto)
            elif r == best_recall and r > 0:
                 pass 

    if not found_feasible:
        print("ADVERTENCIA: Ningún umbral cumple la restricción de precisión. Usando 0.5.")
        best_thresh = 0.5
    else:
        print(f"Umbral Óptimo Encontrado: {best_thresh:.4f}")
        print(f"Métricas Esperadas -> Recall: {best_recall:.4f} | Precision: {final_precision:.4f}")

    # Gráfico de Trade-off
    plt.figure(figsize=(10, 6))
    plt.plot(umbrales, precision_list, label='Precision', color='blue')
    plt.plot(umbrales, recall_list, label='Recall', color='green')
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Optimum ({best_thresh:.2f})')
    plt.axhline(PRECISION_MINIMA, color='gray', linestyle=':', label='Min Precision')
    plt.xlabel('Probabilidad de Corte')
    plt.title('Curvas de Operación para Calibración de Umbral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_thresh

def paso_6_evaluacion_final(modelo, X_test, y_test, umbral, feature_names):
    """
    Evaluación final utilizando el umbral calibrado.
    """
    print("\n--- Validación Final en Test Set ---")
    
    # Probabilidades del GPC (naturalmente bien calibradas)
    probs_test = modelo.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= umbral).astype(int)
    
    print(classification_report(y_test, preds_test, target_names=["OK", "DEFECTO"]))
    
    cm = confusion_matrix(y_test, preds_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Pred OK", "Pred Defecto"], 
                yticklabels=["Real OK", "Real Defecto"])
    plt.title(f'Matriz de Confusión (Umbral {umbral:.3f})')
    plt.show()
    
    # Guardado
    artefactos = {
        "modelo_pipeline": modelo,
        "umbral_optimo": umbral,
        "feature_names": feature_names
    }
    with open('modelo_GPC_ARD_soldadura.pkl', 'wb') as f:
        pickle.dump(artefactos, f)
    print("Modelo GPC guardado exitosamente.")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # 1. Prepara datos
    try:
        X, y = paso_1_cargar_y_preparar_datos(FEATURE_NAMES)
    except NameError:
        print("Por favor, define las funciones de extracción de características (copiar del script anterior).")
        return

    if X is None: return

    # 2. Dividir
    X_train, X_test, y_train, y_test = paso_2_dividir_datos(X, y)
    
    # ---------------------------------------------------------
    # 2. RONDA 1: ENTRENAMIENTO DE DESCUBRIMIENTO (32 FEATURES)
    # ---------------------------------------------------------
    print("\n🔴 RONDA 1: Entrenando con TODAS las variables para descubrir relevancia...")
    modelo_exploratorio = paso_3_entrenar_svm(X_train, y_train)
    
    # ---------------------------------------------------------
    # 3. SELECCIÓN DE VARIABLES (ARD)
    # ---------------------------------------------------------
    print("\n🔍 ANALIZANDO QUÉ VARIABLES IMPORTAN...")
    
    # CAMBIO 2: Lógica de selección completamente nueva.
    # Como el SVM no tiene ARD, usamos "Permutation Importance":
    # Desordenamos aleatoriamente cada columna y vemos cuánto cae el Recall.
    # Si el recall cae mucho, la variable era importante.
    
    result = permutation_importance(
        modelo_exploratorio, X_train, y_train, 
        n_repeats=10, 
        random_state=42, 
        n_jobs=-1, 
        scoring='recall' # Nos importa qué variables afectan al Recall
    )
    
    # Normalizamos la importancia (0 a 100)
    importancia_cruda = result.importances_mean
    # Evitamos negativos (variables que al romperlas mejoran el modelo, son ruido puro)
    importancia_cruda = np.maximum(importancia_cruda, 0) 
    
    if np.max(importancia_cruda) > 0:
        relevancia_norm = 100 * (importancia_cruda / np.max(importancia_cruda))
    else:
        relevancia_norm = np.zeros_like(importancia_cruda)

    # CRITERIO: Nos quedamos con las que tengan > 5% de importancia relativa
    # (Bajamos un poco el umbral respecto a GPC porque la permutación es más estricta)
    indices_vip = np.where(relevancia_norm >= 5.0)[0]
    features_vip = np.array(FEATURE_NAMES)[indices_vip]
    
    # Gráfico rápido de texto
    df_imp = pd.DataFrame({'Feature': FEATURE_NAMES, 'Imp': relevancia_norm})
    df_imp = df_imp.sort_values(by='Imp', ascending=False)
    print("\nTop 10 Variables más importantes (SVM):")
    print(df_imp.head(10))

    if len(features_vip) == 0:
        print("⚠ Advertencia: Ninguna variable superó el umbral. Usando top 5.")
        indices_vip = df_imp.index[:5].to_numpy()
        features_vip = np.array(FEATURE_NAMES)[indices_vip]

    print(f"\n✅ SELECCIONADAS {len(features_vip)} VARIABLES CRÍTICAS:")
    print(list(features_vip))
    
    # ---------------------------------------------------------
    # 4. PREPARAR DATOS VIP
    # ---------------------------------------------------------
    # Filtramos los datasets para quedarnos solo con las columnas VIP
    # IMPORTANTE: .iloc para filtrar por posición de columna
    X_train_vip = X_train.iloc[:, indices_vip]
    X_test_vip  = X_test.iloc[:, indices_vip]

    # ---------------------------------------------------------
    # 5. RONDA 2: ENTRENAMIENTO FINAL (SOLO FEATURES VIP)
    # ---------------------------------------------------------
    print("\n🟢 RONDA 2: Re-entrenando modelo FINAL (Ultrarrobusto)...")
    
    # ¡REUTILIZAMOS LA MISMA FUNCIÓN DE ENTRENAMIENTO!
    # Como X_train_vip tiene menos columnas, la función se adapta sola.
    modelo_final = paso_3_entrenar_svm(X_train_vip, y_train)

    # ---------------------------------------------------------
    # 6. OPTIMIZACIÓN DE UMBRAL Y EVALUACIÓN (CON MODELO FINAL)
    # ---------------------------------------------------------
    # Usamos el modelo final y los datos reducidos
    umbral_optimo = paso_5_optimizar_umbral_probabilistico(modelo_final, X_train_vip, y_train)
    
    paso_6_evaluacion_final(modelo_final, X_test_vip, y_test, umbral_optimo, features_vip)
    
if __name__ == "__main__":
    main()