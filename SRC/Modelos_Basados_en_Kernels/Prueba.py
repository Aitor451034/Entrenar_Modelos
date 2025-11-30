"""
Script de Detección de Defectos en Soldadura (RSW) - VERSIÓN ULTRAROBUSTA
Arquitectura: SVM (Kernel RBF) con Selección Secuencial (SFS)
Evaluación: Estándar Robusto (Gráficos, Umbrales, Análisis de Errores y Bias-Varianza)
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
    train_test_split, GridSearchCV, cross_val_predict, RepeatedStratifiedKFold, StratifiedKFold, learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    fbeta_score, make_scorer, classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score, roc_curve, auc
)

# --- SVM y Selección Avanzada ---
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance # Necesario para interpretar SVM RBF

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

TEST_SIZE_RATIO = 0.40      
RANDOM_STATE_SEED = 42      
N_SPLITS_CV = 5             
FBETA_BETA = 2              
PRECISION_MINIMA = 0.75     

# ==============================================================================
# 3. FUNCIONES DE PREPROCESAMIENTO (INVARIANTES)
# ==============================================================================
def leer_archivo():
    print("Selecciona el archivo CSV que contiene los datos...")
    root = tk.Tk()
    root.withdraw()
    ruta_csv = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
    if not ruta_csv: return None
    try:
        df = pd.read_csv(ruta_csv, encoding="utf-8", sep=";", on_bad_lines="skip", 
                         header=None, decimal=",", skiprows=3)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def calcular_pendiente(resistencias, tiempos):
    if len(resistencias) <= 1 or len(tiempos) <= 1: return [0]
    pendientes = []
    for i in range(len(resistencias) - 1):
        delta_t = tiempos[i + 1] - tiempos[i]
        delta_r = resistencias[i + 1] - resistencias[i]
        pendiente_actual = 0 if delta_t == 0 else (delta_r / delta_t) * 100
        pendientes.append(round(np.nan_to_num(pendiente_actual, nan=0), 2))
    return pendientes

def calcular_derivadas(resistencias, tiempos):
    if len(resistencias) <= 1 or len(tiempos) <= 1:
        return np.array([0]), np.array([0]), np.array([0])
    primera = np.gradient(resistencias, tiempos)
    segunda = np.gradient(primera, tiempos)
    tercera = np.gradient(segunda, tiempos)
    return np.nan_to_num(primera, nan=0), np.nan_to_num(segunda, nan=0), np.nan_to_num(tercera, nan=0)

def preprocesar_dataframe_inicial(df):
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
            try: df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except: pass
    return new_df

def extraer_features_fila_por_fila(new_df):
    X_calculado = []
    y_calculado = []
    print(f"Iniciando cálculo de features para {len(new_df)} puntos de soldadura...")

    for i in new_df.index:
        datos_voltaje = new_df.loc[i, "Voltajes inst."]
        datos_corriente = new_df.loc[i, "Corrientes inst."]
        if pd.isna(datos_voltaje) or pd.isna(datos_corriente): continue
            
        valores_voltaje = [round(float(v), 0) for v in datos_voltaje.split(';') if v.strip()]
        valores_corriente = [round((0.001 if float(v) == 0 else float(v)), 0) for v in datos_corriente.split(';') if v.strip()]

        valores_resistencia = [round(v / c, 2) if c != 0 else 0 for v, c in zip(valores_voltaje, valores_corriente)]
        valores_resistencia.append(0)
        
        ns = int(new_df.loc[i, "Ns"])
        ts2 = int(new_df.loc[i, "Ts2"])
        t_soldadura = (np.linspace(0, ts2, ns + 1)).tolist()

        min_len = min(len(t_soldadura), len(valores_resistencia), len(valores_voltaje), len(valores_corriente))
        t_soldadura = t_soldadura[:min_len]
        valores_resistencia = valores_resistencia[:min_len]
        valores_voltaje = valores_voltaje[:min_len]
        valores_corriente = valores_corriente[:min_len]
        
        if not t_soldadura: continue

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
        
        # Energía y Features
        arr_v = np.array(valores_voltaje) / 100.0
        arr_i_amp = np.array(valores_corriente) * 10
        arr_t_sec = np.array(t_soldadura) / 1000.0
        q = np.nan_to_num(np.trapz(arr_v * arr_i_amp, x=arr_t_sec), nan=0)
        
        area_bajo_curva = np.nan_to_num(np.trapz(valores_resistencia, t_soldadura), nan=0)
        resistencia_ultima = valores_resistencia[-2]
        
        k4 = 0 if (t_e - t_R_max) == 0 else ((r_e - resistencia_max) / (t_e - t_R_max)) * 100
        k3 = 0 if (t_R_max - t0) == 0 else ((resistencia_max - r0) / (t_R_max - t0)) * 100
        
        desv = np.nan_to_num(np.std(valores_resistencia), nan=0)
        rms = np.nan_to_num(np.sqrt(np.mean(np.square(valores_resistencia))), nan=0)
        rango_tiempo_max_min = np.nan_to_num(t_R_max - t_soldadura_min, nan=0)
        rango_rmax_rmin = np.nan_to_num(resistencia_max - resistencia_min, nan=0)
        
        voltaje_max = max(valores_voltaje)
        t_voltaje_max = t_soldadura[np.argmax(valores_voltaje)]
        voltaje_final = valores_voltaje[-2]
        t_voltaje_final = t_soldadura[-2]
        pendiente_V = 0 if (t_voltaje_max - t_voltaje_final) == 0 else (voltaje_max - voltaje_final) / (t_voltaje_max - t_voltaje_final)
        
        r_mean_post_max = np.nan_to_num(np.mean(valores_resistencia[I_R_max:]), nan=0)
        resistencia_inicial = np.nan_to_num(r0, nan=2000)
        r_mean = np.nan_to_num(np.mean(valores_resistencia[:-1]), nan=0)
        rango_r_beta_alfa = np.nan_to_num(resistencia_max - r0, nan=0)
        rango_r_e_beta = np.nan_to_num(r_e - resistencia_max, nan=0)
        rango_t_e_beta = np.nan_to_num(t_e - t_R_max, nan=0)
        desv_R = np.nan_to_num(np.std(valores_resistencia[:I_R_max]), nan=0)
        
        pendientes = calcular_pendiente(valores_resistencia, t_soldadura)
        pendientes_negativas_post = sum(1 for p in pendientes[I_R_max:] if p < 0)
        
        area_pre_mitad = np.nan_to_num(np.trapz(valores_resistencia[:I_R_max + 1], t_soldadura[:I_R_max + 1]), nan=0)
        area_post_mitad = np.nan_to_num(np.trapz(valores_resistencia[I_R_max:], t_soldadura[I_R_max:]), nan=0)
        desv_pre_mitad_t = np.nan_to_num(np.std(valores_resistencia[:I_R_max + 1]), nan=0)
        
        d1, d2, d3 = calcular_derivadas(valores_resistencia, t_soldadura)
        max_curvatura = np.nan_to_num(np.max(np.abs(d2)), nan=0)
        num_puntos_inflexion = np.nan_to_num(len(np.where(np.diff(np.sign(d2)))[0]), nan=0)
        max_jerk = np.nan_to_num(np.max(np.abs(d3)), nan=0)
        
        mediana = np.nan_to_num(np.median(valores_resistencia), nan=0)
        varianza = np.nan_to_num(np.var(valores_resistencia), nan=0)
        rango_intercuartilico = np.nan_to_num((np.percentile(valores_resistencia, 75) - np.percentile(valores_resistencia, 25)), nan=0)
        asimetria = np.nan_to_num(skew(valores_resistencia), nan=0)
        curtosis = np.nan_to_num(kurtosis(valores_resistencia), nan=0)
        
        picos, _ = find_peaks(np.array(valores_resistencia), height=0)
        valles, _ = find_peaks(-np.array(valores_resistencia))
        num_picos = len(picos)
        num_valles = len(valles)
        
        t_mean = np.mean(t_soldadura)
        r_mean_ols = np.mean(valores_resistencia)
        num_m = sum((r_mean_ols - ri) * (t_mean - ti) for ri, ti in zip(valores_resistencia, t_soldadura))
        den_m = sum((r_mean_ols - ri) ** 2 for ri in valores_resistencia)
        m_min_cuadrados = 0 if den_m == 0 else num_m / den_m

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
# 4. FUNCIONES DE UTILERÍA Y GRÁFICOS
# ==============================================================================
def graficar_distribucion_energia(X, y):
    print("\n--- Generando Gráfico de Diagnóstico de Energía (q) ---")
    plt.figure(figsize=(10, 6))
    try:
        col_q = X['q'] if isinstance(X, pd.DataFrame) else X[:, 30]
        sns.histplot(col_q[y == 0], color='green', label='OK', kde=True, element="step")
        sns.histplot(col_q[y == 1], color='red', label='Defecto', kde=True, element="step")
        plt.title('DIAGNÓSTICO: Distribución de Energía (Joules)')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"No se pudo generar el gráfico de energía: {e}")

def _plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=["Pred: OK", "Pred: Defecto"],
        yticklabels=["Real: OK", "Real: Defecto"]
    )
    plt.title(title)
    plt.ylabel('Realidad')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 5. PIPELINE DE APRENDIZAJE ROBUSTO
# ==============================================================================

def paso_1_cargar_y_preparar_datos(feature_names):
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

def paso_3_seleccion_secuencial_sfs(X_train, y_train, feature_names):
    """
    Selección secuencial (SFS).
    """
    print("\n" + "="*60)
    print("--- PASO 3: SELECCIÓN AUTOMÁTICA (SFS) ---")
    print("="*60)

    svm_selector = Pipeline([
        ('scaler', RobustScaler()),
        ('svm', SVC(kernel='rbf', C=50, gamma='auto', class_weight='balanced', random_state=RANDOM_STATE_SEED))
    ])
    
    sfs = SequentialFeatureSelector(
        svm_selector,
        n_features_to_select="auto", 
        tol=0.005,      
        direction='forward',
        scoring=make_scorer(fbeta_score, beta=FBETA_BETA), 
        cv=StratifiedKFold(n_splits=5), 
        n_jobs=-1
    )
    
    start_time = time.time()
    sfs.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    indices_vip = np.where(sfs.get_support())[0]
    features_vip = np.array(feature_names)[indices_vip]
    
    print(f"\nSelección completada en {elapsed_time:.2f} segundos.")
    print(f"✅ Variables seleccionadas ({len(features_vip)}): {list(features_vip)}")
    
    return features_vip, indices_vip

def paso_4_visualizar_importancia_sfs_permutation(modelo, X, y, feature_names):
    """
    MODIFICADO: Genera el gráfico de barras horizontales usando Permutation Importance.
    SVM con RBF no tiene 'coef_', por lo que usamos permutación para ver qué variable 
    afecta más al modelo si la alteramos.
    """
    print("\n--- Generando Gráfico de Importancia (Permutation Importance) ---")
    
    # Calculamos la importancia permutando características en el set de validación
    # Esto nos dice: "¿Cuánto cae el Score si mezclo aleatoriamente esta columna?"
    r = permutation_importance(
        modelo, X, y,
        n_repeats=10,
        random_state=RANDOM_STATE_SEED,
        scoring=make_scorer(fbeta_score, beta=FBETA_BETA),
        n_jobs=-1
    )
    
    importancias = r.importances_mean
    sorted_idx = importancias.argsort()

    # Crear DataFrame para visualización
    df_imp = pd.DataFrame({
        'Feature': np.array(feature_names)[sorted_idx],
        'Importance': importancias[sorted_idx]
    })
    
    # Gráfico estilo CatBoost (Barras horizontales)
    plt.figure(figsize=(10, 8))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='teal')
    plt.xlabel("Importancia de Permutación (Caída en F2-Score)")
    plt.title("Importancia de Variables (SVM RBF - Permutación)")
    plt.tight_layout()
    plt.show()

def paso_5_entrenar_final_y_optimizar_umbral(X_train, y_train, n_splits, precision_minima):
    """
    Entrena el modelo final y optimiza el umbral con gráficas P/R vs Threshold.
    """
    print(f"\n--- PASO 5: Entrenamiento Final y Optimización de Umbral ---")
    
    # 1. Pipeline Final
    pipeline = Pipeline([
        ('scaler', RobustScaler()), 
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE_SEED))
    ])

    # 2. GridSearch Fino
    param_grid = {
        'svm__C': [1, 10, 50, 100, 500],
        'svm__gamma': ['scale', 0.1, 0.01]
    }
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE_SEED)
    f2_scorer = make_scorer(fbeta_score, beta=FBETA_BETA)

    search = GridSearchCV(pipeline, param_grid, cv=rskf, scoring=f2_scorer, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    mejor_modelo = search.best_estimator_
    
    print(f"Mejor CV Score: {search.best_score_:.4f}")
    
    # 3. Optimización de Umbral (Estilo CatBoost)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE_SEED)
    y_probas_cv = cross_val_predict(mejor_modelo, X_train, y_train, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]

    umbrales = np.linspace(0.01, 0.99, 1000)
    best_recall = -1
    optimal_threshold = 0.5
    best_precision = 0

    precision_vals, recall_vals = [], []

    for thresh in umbrales:
        y_pred = (y_probas_cv >= thresh).astype(int)
        prec = precision_score(y_train, y_pred, zero_division=0)
        rec = recall_score(y_train, y_pred, zero_division=0)
        precision_vals.append(prec)
        recall_vals.append(rec)

        if prec >= precision_minima:
            if rec > best_recall:
                best_recall = rec
                optimal_threshold = thresh
                best_precision = prec
            elif rec == best_recall:
                optimal_threshold = min(optimal_threshold, thresh)

    if best_recall == -1:
        print(f"⚠ Ningún umbral cumple Precision >= {precision_minima}. Usando 0.5.")
        optimal_threshold = 0.5
    else:
        print(f"✅ Umbral Óptimo: {optimal_threshold:.4f} (Recall: {best_recall:.4f}, Prec: {best_precision:.4f})")

    # Gráfico Precision/Recall vs Umbral
    plt.figure(figsize=(10, 5))
    plt.plot(umbrales, precision_vals, label='Precision', color='blue')
    plt.plot(umbrales, recall_vals, label='Recall', color='green')
    plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Óptimo: {optimal_threshold:.3f}')
    plt.axhline(precision_minima, color='gray', linestyle=':', label='Min Precision')
    plt.title('Evolución de Precision y Recall según Umbral (CV)')
    plt.xlabel('Umbral')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return mejor_modelo, optimal_threshold

def paso_6_evaluacion_final_robusta(mejor_modelo, X_test, y_test, optimal_threshold, feature_names):
    """
    Evaluación completa (Reporte, ROC, Matriz, Análisis de Errores).
    """
    print("\n--- PASO 6: Evaluación Final en Test Set ---")
    
    # Predicciones
    y_probas_test = mejor_modelo.predict_proba(X_test)[:, 1]
    preds_test_optimas = (y_probas_test >= optimal_threshold).astype(int)

    # Reporte
    print(f"\nReporte de Clasificación (Umbral {optimal_threshold:.4f}):")
    print(classification_report(y_test, preds_test_optimas, target_names=['Sin Defecto', 'Pegado']))

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_probas_test)
    auc_test = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC Test = {auc_test:.4f}', color='purple', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title('Curva ROC - Test Set')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    # Matriz Confusión
    cm = confusion_matrix(y_test, preds_test_optimas)
    _plot_confusion_matrix(cm, f"Matriz Test (Umbral {optimal_threshold:.4f})")

    # Análisis de Errores
    print("\n--- Análisis de Errores Detallado ---")
    df_analisis = pd.DataFrame({'Real': y_test, 'Probabilidad': y_probas_test, 'Prediccion': preds_test_optimas})
    
    # Falsos Negativos
    fn = df_analisis[(df_analisis['Real'] == 1) & (df_analisis['Prediccion'] == 0)]
    print(f"\n[Falsos Negativos] Defectos NO detectados: {len(fn)}")
    if not fn.empty: print(fn.head(10).to_string())

    # Falsos Positivos
    fp = df_analisis[(df_analisis['Real'] == 0) & (df_analisis['Prediccion'] == 1)]
    print(f"\n[Falsos Positivos] Falsas Alarmas: {len(fp)}")
    if not fp.empty: print(fp.head(10).to_string())

    # Guardado
    artefactos = {
        "pipeline_completo": mejor_modelo,
        "umbral": optimal_threshold,
        "feature_names": feature_names
    }
    with open('modelo_svm_sfs_optimizado.pkl', 'wb') as f:
        pickle.dump(artefactos, f)
    print("\nModelo guardado exitosamente.")

def paso_extra_graficar_bias_varianza(modelo, X, y, cv, scoring_metric='f1'):
    print("\nGenerando Curvas de Aprendizaje...")
    train_sizes, train_scores, val_scores = learning_curve(
        modelo, X, y, cv=cv, scoring=scoring_metric, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Score Entrenamiento")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.plot(train_sizes, val_mean, 'o-', color="g", label="Score Validación (CV)")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
    
    plt.title('Curva de Aprendizaje (Bias vs Varianza)')
    plt.xlabel('Tamaño Training Set')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()

    gap = train_mean[-1] - val_mean[-1]
    print(f"Gap Final: {gap:.4f}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # 1. Carga
    try: X, y = paso_1_cargar_y_preparar_datos(FEATURE_NAMES)
    except NameError: return
    if X is None: return

    # Diagnóstico Energía
    graficar_distribucion_energia(X, y)

    # 2. División
    X_train, X_test, y_train, y_test = paso_2_dividir_datos(X, y)
    
    # 3. Selección SFS
    features_vip, indices_vip = paso_3_seleccion_secuencial_sfs(X_train, y_train, FEATURE_NAMES)
    
    X_train_vip = X_train.iloc[:, indices_vip]
    X_test_vip  = X_test.iloc[:, indices_vip]

    # 4. Entrenamiento Final y Optimización Umbral
    modelo_final, umbral_optimo = paso_5_entrenar_final_y_optimizar_umbral(
        X_train_vip, y_train, N_SPLITS_CV, PRECISION_MINIMA
    )

    # 4b. Visualizar Importancia (Permutation) - Ahora que tenemos el modelo final
    paso_4_visualizar_importancia_sfs_permutation(modelo_final, X_train_vip, y_train, features_vip)

    # 5. Evaluación Final
    paso_6_evaluacion_final_robusta(
        modelo_final, X_test_vip, y_test, umbral_optimo, features_vip
    )

    # 6. Bias-Varianza
    cv_plot = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE_SEED)
    f2_scorer = make_scorer(fbeta_score, beta=FBETA_BETA)
    paso_extra_graficar_bias_varianza(modelo_final, X_train_vip, y_train, cv_plot, f2_scorer)

if __name__ == "__main__":
    main()