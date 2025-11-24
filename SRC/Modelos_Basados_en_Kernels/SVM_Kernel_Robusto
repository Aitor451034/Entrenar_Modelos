"""
Script de Detección de Defectos en Soldadura (RSW) - VERSIÓN ULTRAROBUSTA
Arquitectura: SVM (Kernel RBF) con Búsqueda Secuencial de Características (SFS)
Optimización: Prioridad Recall, Escalado Robusto y GridSearch Extensivo.
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
    train_test_split, GridSearchCV, cross_val_predict, RepeatedStratifiedKFold, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler es mejor para outliers
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    fbeta_score, make_scorer, classification_report, confusion_matrix,
    precision_score, recall_score,roc_auc_score,roc_curve
)

# --- SVM y Selección Avanzada ---
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector # El método más robusto (y lento)
from sklearn.inspection import permutation_importance

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

# Parámetros del experimento
TEST_SIZE_RATIO = 0.30      # 30% para test final
RANDOM_STATE_SEED = 42      # Semilla para reproducibilidad
N_SPLITS_CV = 5             # Número de carpetas para validación cruzada (ESTA FALTABA)
FBETA_BETA = 2              # Prioridad Recall (F2-Score)
PRECISION_MINIMA = 0.75     # Restricción de precisión mínima deseada

# ==============================================================================
# 3. FUNCIONES DE PREPROCESAMIENTO (MISMAS QUE TU SCRIPT ORIGINAL)
# ==============================================================================
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
# 4. PIPELINE DE APRENDIZAJE ROBUSTO
# ==============================================================================

def paso_1_cargar_y_preparar_datos(feature_names):
    # (Misma lógica de carga)
    df_raw = leer_archivo()
    if df_raw is None: return None, None
    try:
        df_pre = preprocesar_dataframe_inicial(df_raw)
        X_arr, y_arr = extraer_features_fila_por_fila(df_pre)
    except NameError:
        print("Error: Funciones de extracción no definidas.")
        return None, None

    if X_arr.size == 0: return None, None
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="Etiqueta_Defecto")
    print(f"Datos cargados: {len(X)} muestras. Tasa defecto: {y.mean():.2%}")
    return X, y

def paso_2_dividir_datos(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, 
        random_state=RANDOM_STATE_SEED, stratify=y
    )

def paso_3_entrenar_svm_exhaustivo(X_train, y_train):
    """
    Entrenamiento SVM diseñado para generalización máxima.
    - RobustScaler: Ignora outliers extremos de soldadura.
    - Grid Ampliado: Busca en escala logarítmica fina.
    """
    print(f"\n--- Entrenando SVM Exhaustivo ({X_train.shape[1]} vars) ---")
    
    # Usamos RobustScaler en lugar de Standard para aguantar picos de sensores
    pipeline = Pipeline([
        ('scaler', RobustScaler()), 
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE_SEED))
    ])

    # Grid Logarítmico Extendido
    # C bajo = 0.1 (Mucho sesgo, poco variance) -> Generaliza mejor
    # C alto = 1000 (Poco sesgo, mucho variance) -> Ajuste fino
    param_grid = {
        'svm__C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        'svm__gamma': ['scale', 'auto', 0.1, 0.05, 0.01, 0.005, 0.001]
    }

    # CV más rigurosa (5 splits x 5 repeticiones)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE_SEED)
    f2_scorer = make_scorer(fbeta_score, beta=FBETA_BETA)

    search = GridSearchCV(
        pipeline, param_grid, cv=rskf, scoring=f2_scorer, n_jobs=-1, verbose=1
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"  -> Mejor F2 (CV): {search.best_score_:.4f}")
    print(f"  -> Params Óptimos: {search.best_params_}")
    print(f"  -> Tiempo GridSearch: {elapsed_time:.2f}s")
    
    return search.best_estimator_

def paso_4_seleccion_secuencial_sfs(X_train, y_train, feature_names):
    """
    SFS (Sequential Feature Selector) en MODO AUTOMÁTICO.
    Ya no le decimos "dame 12". Le decimos "sigue añadiendo variables hasta
    que el F2 Score deje de mejorar significativamente".
    """
    print("\n" + "="*60)
    print("--- PASO 4: SELECCIÓN AUTOMÁTICA (SFS AUTO-STOP) ---")
    print("El algoritmo decidirá el número óptimo de variables...")
    print("="*60)

    # Estimador base más estricto para la selección
    svm_selector = Pipeline([
        ('scaler', RobustScaler()),
        ('svm', SVC(
            kernel='rbf', 
            C=50,         # Penalización alta para buscar patrones fuertes
            gamma='auto', 
            class_weight='balanced', 
            random_state=RANDOM_STATE_SEED
        ))
    ])
    
    # CONFIGURACIÓN AUTOMÁTICA
    # tol=0.003 -> Si la nueva variable no mejora el F2 al menos un 0.3%, se detiene.
    # Esto previene agregar "basura" que solo mejora decimales irrelevantes.
    sfs = SequentialFeatureSelector(
        svm_selector,
        n_features_to_select="auto", 
        tol=0.005,      
        direction='forward',
        scoring=make_scorer(fbeta_score, beta=FBETA_BETA), 
        cv=StratifiedKFold(n_splits=5), # 5 Splits para robustez
        n_jobs=-1
    )
    
    start_time = time.time()
    sfs.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    indices_vip = np.where(sfs.get_support())[0]
    features_vip = np.array(feature_names)[indices_vip]
    
    print(f"\nSelección completada en {elapsed_time:.2f} segundos.")
    print(f"✅ El algoritmo determinó que el NÚMERO ÓPTIMO de variables es: {len(features_vip)}")
    print(f"Variables seleccionadas: {list(features_vip)}")
    
    return features_vip, indices_vip

def paso_5_optimizar_umbral(mejor_modelo, X_train, y_train, n_splits, precision_minima, random_state):
    """
    Busca el umbral óptimo usando predicciones de Validación Cruzada (Cross-Validation).
    Muestra gráficas de rendimiento (ROC, CM 0.5, CM Óptima) basadas en estos datos de validación.
    """
    print(f"\n--- PASO 5: Optimización y Análisis en Validación Cruzada (CV) ---")
    print(f"Objetivo: Maximizar Recall sujeto a Precision >= {precision_minima}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print("Generando predicciones de validación (Out-of-Fold)...")
    # Estas probabilidades son las generadas cuando cada fold actuó como set de validación
    y_probas_cv = cross_val_predict(
        mejor_modelo,
        X_train,
        y_train,
        cv=skf,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    # --- 1. Lógica de Optimización del Umbral ---
    lista_umbrales = np.linspace(0.01, 0.99, 1000)
    best_recall = -1
    optimal_threshold = 0.5
    best_precision = 0

    precision_vals, recall_vals = [], []

    for thresh in lista_umbrales:
        y_pred_temp = (y_probas_cv >= thresh).astype(int)
        prec = precision_score(y_train, y_pred_temp, zero_division=0)
        rec = recall_score(y_train, y_pred_temp, zero_division=0)
        
        precision_vals.append(prec)
        recall_vals.append(rec)

        if prec >= precision_minima:
            if rec > best_recall:
                best_recall = rec
                optimal_threshold = thresh
                best_precision = prec
            elif rec == best_recall:
                # Si empate en recall, buscamos el umbral más bajo (más seguro para defectos)
                # o el que tenga mejor precisión. Aquí priorizamos umbral bajo.
                optimal_threshold = min(optimal_threshold, thresh)

    if best_recall == -1:
        print(f"⚠ ADVERTENCIA: Ningún umbral cumple Precision >= {precision_minima}. Se usará 0.5.")
        optimal_threshold = 0.5
    else:
        print(f"✅ Umbral Óptimo Encontrado: {optimal_threshold:.4f}")
        print(f"   Métricas en CV -> Recall: {best_recall:.4f} | Precision: {best_precision:.4f}")

    # --- 2. Gráficos de Rendimiento en Validación (CV) ---
    
    # A) Curva ROC (Validación)
    fpr, tpr, _ = roc_curve(y_train, y_probas_cv)
    auc_val = roc_auc_score(y_train, y_probas_cv)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC Validación = {auc_val:.4f}', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - Datos de Validación Cruzada')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    # B) Comparación de Matrices de Confusión (0.5 vs Óptimo) en Validación
    print("\n--- Comparativa en Validación Cruzada ---")
    
    # Predicciones binarias con umbral 0.5
    preds_cv_defecto = (y_probas_cv >= 0.5).astype(int)
    cm_defecto = confusion_matrix(y_train, preds_cv_defecto)
    _plot_confusion_matrix(cm_defecto, "Matriz CV (Umbral Estándar 0.5)")

    # Predicciones binarias con umbral óptimo
    preds_cv_optimo = (y_probas_cv >= optimal_threshold).astype(int)
    cm_optimo = confusion_matrix(y_train, preds_cv_optimo)
    _plot_confusion_matrix(cm_optimo, f"Matriz CV (Umbral Óptimo {optimal_threshold:.4f})")

    # C) Gráfico Precision-Recall vs Umbral
    plt.figure(figsize=(10, 5))
    plt.plot(lista_umbrales, precision_vals, label='Precision', color='blue')
    plt.plot(lista_umbrales, recall_vals, label='Recall', color='green')
    plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Óptimo: {optimal_threshold:.3f}')
    plt.axhline(precision_minima, color='gray', linestyle=':', label=f'Min Prec: {precision_minima}')
    plt.xlabel('Umbral')
    plt.title('Evolución de Precision y Recall según Umbral (CV)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return optimal_threshold

def paso_6_evaluacion_final_y_guardado(mejor_modelo, X_test, y_test, scaler, optimal_threshold, feature_names):
    """
    Evaluación final en el Test Set (datos nunca vistos).
    Muestra ROC, CM (0.5 y Óptima) y guarda el modelo.
    """
    print("\n--- PASO 6: Evaluación Final en Conjunto de Prueba (Test Set) ---")
    
    # Obtener probabilidades reales del modelo final
    y_probas_test = mejor_modelo.predict_proba(X_test)[:, 1]
    
    # --- 1. Reporte de Clasificación (con Umbral Óptimo) ---
    preds_test_optimas = (y_probas_test >= optimal_threshold).astype(int)
    print(f"\nReporte de Clasificación (Test Set - Umbral {optimal_threshold:.4f}):")
    print(classification_report(y_test, preds_test_optimas, target_names=['Sin Defecto', 'Pegado']))

    # --- 2. Curva ROC (Test Set) ---
    fpr, tpr, _ = roc_curve(y_test, y_probas_test)
    auc_test = roc_auc_score(y_test, y_probas_test)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC Test = {auc_test:.4f}', color='purple', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Conjunto de Prueba (Test)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    # --- 3. Matrices de Confusión (Test Set) ---
    # A) Umbral por defecto 0.5
    preds_test_defecto = (y_probas_test >= 0.5).astype(int)
    cm_test_defecto = confusion_matrix(y_test, preds_test_defecto)
    _plot_confusion_matrix(cm_test_defecto, "Matriz Test (Umbral Estándar 0.5)")
    
    # B) Umbral Óptimo
    cm_test_optimo = confusion_matrix(y_test, preds_test_optimas)
    _plot_confusion_matrix(cm_test_optimo, f"Matriz Test (Umbral Óptimo {optimal_threshold:.4f})")

    # --- 4. Análisis de Errores (Basado en el óptimo) ---
    print("\n--- Análisis de Errores (Test Set - Umbral Óptimo) ---")
    df_analisis = pd.DataFrame(y_test).copy() # Copia segura
    df_analisis.columns = ['Etiqueta_Real'] # Renombrar para claridad si es serie
    df_analisis['Probabilidad'] = y_probas_test
    df_analisis['Prediccion'] = preds_test_optimas
    
    # Falsos Negativos (Era Defecto 1, predijo OK 0)
    fn = df_analisis[(df_analisis['Etiqueta_Real'] == 1) & (df_analisis['Prediccion'] == 0)]
    print(f"\n[Falsos Negativos] Defectos NO detectados: {len(fn)}")
    if not fn.empty:
        print(fn.head(10).to_string())

    # Falsos Positivos (Era OK 0, predijo Defecto 1)
    fp = df_analisis[(df_analisis['Etiqueta_Real'] == 0) & (df_analisis['Prediccion'] == 1)]
    print(f"\n[Falsos Positivos] Falsas Alarmas: {len(fp)}")
    if not fp.empty:
        print(fp.head(10).to_string())

    # --- 5. Guardado ---
    print("\nGuardando pipeline y configuración...")
    artefactos = {
        "pipeline_completo": mejor_modelo,
        "umbral": optimal_threshold,
        "feature_names": feature_names
    }
    with open('modelo_catboost_optimizado.pkl', 'wb') as f:
        pickle.dump(artefactos, f)
    print("¡Modelo guardado exitosamente!")

def _plot_confusion_matrix(cm, title):
    """Función auxiliar para graficar."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=["Pred: OK", "Pred: Defecto"],
        yticklabels=["Real: OK", "Real: Defecto"]
    )
    plt.title(title)
    plt.ylabel('Realidad')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.show()
# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # 1. Prepara datos
    try:
        X, y = paso_1_cargar_y_preparar_datos(FEATURE_NAMES)
    except NameError:
        print("Copia las funciones de extracción del script original.")
        return
    if X is None: return

    # 2. Dividir
    X_train, X_test, y_train, y_test = paso_2_dividir_datos(X, y)
    
    # ---------------------------------------------------------
    # 3. SELECCIÓN DE VARIABLES (SFS - ALTO CÓMPUTO)
    # ---------------------------------------------------------
    # Aquí es donde invertimos tiempo de computación para ganar precisión.
    # En lugar de entrenar con todo y luego quitar, usamos SFS para 
    # encontrar el mejor equipo de ~12 variables desde cero.
    
    features_vip, indices_vip = paso_4_seleccion_secuencial_sfs(X_train, y_train, FEATURE_NAMES)
    
    # ---------------------------------------------------------
    # 4. PREPARAR DATOS VIP
    # ---------------------------------------------------------
    X_train_vip = X_train.iloc[:, indices_vip]
    X_test_vip  = X_test.iloc[:, indices_vip]

    # ---------------------------------------------------------
    # 5. ENTRENAMIENTO FINAL (OPTIMIZACIÓN PROFUNDA)
    # ---------------------------------------------------------
    print("\n🟢 Entrenando modelo FINAL sobre variables seleccionadas...")
    modelo_final = paso_3_entrenar_svm_exhaustivo(X_train_vip, y_train)

    # ---------------------------------------------------------
    # 6. OPTIMIZACIÓN DE UMBRAL (VISUALIZACIÓN AVANZADA)
    # ---------------------------------------------------------
    # Ahora pasamos los parámetros de configuración para que pinte las curvas ROC y CM
    umbral_optimo = paso_5_optimizar_umbral(
        mejor_modelo=modelo_final, 
        X_train=X_train_vip, 
        y_train=y_train, 
        n_splits=N_SPLITS_CV, 
        precision_minima=PRECISION_MINIMA, 
        random_state=RANDOM_STATE_SEED
    )
    
    # ---------------------------------------------------------
    # 7. EVALUACIÓN FINAL TEST SET
    # ---------------------------------------------------------
    # Pasamos scaler=None porque el SVM ya tiene el Scaler dentro de su Pipeline
    paso_6_evaluacion_final_y_guardado(
        mejor_modelo=modelo_final, 
        X_test=X_test_vip, 
        y_test=y_test, 
        scaler=None, 
        optimal_threshold=umbral_optimo, 
        feature_names=features_vip
    )
if __name__ == "__main__":
    main()