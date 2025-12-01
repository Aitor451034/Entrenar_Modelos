"""
Script para entrenar un modelo Híbrido (SMOTE + Random Forest)
con el objetivo de detectar puntos de soldadura defectuosos (pegados).

El proceso incluye:
1.  Carga de datos y extracción de 32 características (feature engineering).
2.  Separación de datos en entrenamiento (Train) y prueba (Test) y escalado.
3.  Definición de un pipeline de Imbalanced-learn (ImbPipeline) que:
    a. Aplica SMOTE para sobremuestrear la clase minoritaria.
    b. Entrena un modelo RandomForestClassifier.
4.  Búsqueda exhaustiva de hiperparámetros (GridSearchCV) en el pipeline.
5.  Optimización del umbral de decisión (Regla de Sinergia).
6.  Evaluación final y análisis de errores en el conjunto de prueba (Test set).
7.  Guardado del pipeline completo (SMOTE + Scaler + RF) y el umbral.
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

# --- Componentes de Scikit-learn ---
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_predict
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
PRECISION_MINIMA = 0.68


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
        
# ==============================================================================
        # CÁLCULO DE ENERGÍA DIRECTO (P = V * I) - MÁS ROBUSTO
        # ==============================================================================
        
        # 1. Convertir a Numpy
        arr_v = np.array(valores_voltaje)       # Voltios (crudo)
        arr_i_kA = np.array(valores_corriente)  # kA (crudo)
        arr_t_ms = np.array(t_soldadura)        # ms
        
        # --- BLOQUE DE SEGURIDAD DE TAMAÑOS ---
        min_len = min(len(arr_v), len(arr_i_kA), len(arr_t_ms))
        arr_v = arr_v[:min_len]
        arr_i_kA = arr_i_kA[:min_len]
        arr_t_ms = arr_t_ms[:min_len]
        # --------------------------------------

        # 2. Unidades SI
        arr_i_amp = arr_i_kA * 10   # kA -> Amperios
        arr_t_sec = arr_t_ms / 1000.0   # ms -> Segundos
        arra_volts= arr_v/100.0

        # 3. Potencia Instantánea (Watts)
        # Operación vectorizada directa, sin calcular resistencia intermedia
        potencia_watts = arra_volts * arr_i_amp

        # 4. Energía Total (Julios)
        q_joules = np.trapz(potencia_watts, x=arr_t_sec)
        
        q = np.nan_to_num(q_joules, nan=0)

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

def paso_1_cargar_y_preparar_datos(feature_names):
    """Orquesta la carga de datos y la creación de los DataFrames X e y."""
    df_raw = leer_archivo()
    if df_raw is None:
        return None, None
    df_preprocesado = preprocesar_dataframe_inicial(df_raw)
    X_raw, y_raw = extraer_features_fila_por_fila(df_preprocesado)
    
    if X_raw.size == 0:
        print("No se cargaron datos. Terminando.")
        return None, None
        
    X = pd.DataFrame(X_raw, columns=feature_names)
    X = X.applymap(lambda x: round(x, 4))
    y = pd.Series(y_raw, name="Etiqueta_Defecto")

    print("\n--- Resumen de Datos Cargados ---")
    print(f"Total de muestras: {len(X)}")
    print(f"Distribución de clases (antes de SMOTE):\n{y.value_counts(normalize=True)}")
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
    
    # Devolvemos None en lugar del scaler para no romper la estructura del main, 
    # aunque ya no se use aquí.
    return X_train, X_test, y_train, y_test,None

def paso_3_entrenar_modelo(X_train, y_train, n_splits, fbeta, random_state):
    """
    *** LÓGICA CENTRAL: Pipeline Completo (Scaler + Selector + Balanced RF) ***
    """
    print("Iniciando búsqueda de hiperparámetros para Balanced RandomForest...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f2_scorer = make_scorer(fbeta_score, beta=fbeta)

    # 1. Definir el Pipeline
    # NO necesitamos definir la variable modelo_hibrido_BRF fuera, 
    # lo instanciamos directamente dentro del pipeline para evitar confusiones.
    
    pipeline_BRF = ImbPipeline([
        ('scaler', StandardScaler()),           # 1. Escalar
        ('selector', RFE(           # 2. Seleccionar Features
            RandomForestClassifier(n_estimators=400, random_state=random_state, n_jobs=-1,),
            step=0.1,  # Elimina 1 a 1 (máxima precisión)
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

    # *** CORRECCIÓN ***: Título del print
    print("\nImportancia de las 32 características (features) para el modelo Balanced RandomForest:")
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
    # *** CORRECCIÓN ***: Título del gráfico
    titulo = f"Matriz de Confusión - Balanced RandomForest (Umbral Óptimo = {optimal_threshold:.4f})"
    _plot_confusion_matrix(matriz_confusion_opt, titulo)

    # --- 3. Curva ROC ---
    # 
    # *** CORRECCIÓN CRÍTICA ***: Usar probabilidades, no predicciones binarias.
    fpr, tpr, _ = metrics.roc_curve(y_test, predicciones_test_proba)
    auc_score = metrics.roc_auc_score(y_test, predicciones_test_proba)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"Balanced RandomForest (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Clasificador Aleatorio (AUC = 0.5)")
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC (Test Set)')
    plt.legend(loc='lower right')
    plt.grid()
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
    
def paso_extra_graficar_bias_varianza(modelo, X, y, cv, scoring_metric='f1'):
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
    scoring_metric : Métrica a evaluar (f1, recall, precision, etc.)
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
    plt.title(f"Curva de Aprendizaje (Bias vs Varianza) - {scoring_metric}")
    plt.xlabel("Tamaño del Set de Entrenamiento (muestras)")
    plt.ylabel("Score (0 a 1)")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    # ==== INTERPRETACIÓN AUTOMÁTICA DEL GAP TRAIN-VALIDACIÓN ====
    # El gap final indica si hay overfitting:
    # - Gap pequeño (< 0.05) → Buen balance entre bias y varianza
    # - Gap grande (> 0.1) → Posible overfitting, el modelo memorizó entrenamiento
    gap_final = train_mean[-1] - val_mean[-1]
    print(f"Gap final entre Train y Validación: {gap_final:.4f}")
    
    if gap_final < 0.05:
        print("✓ Gap pequeño: El modelo tiene buen balance (sin overfitting notable)")
    elif gap_final < 0.10:
        print("⚠ Gap moderado: Cierto overfitting, pero aceptable")
    else:
        print("✗ Gap grande: Overfitting significativo, considera regularización")

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

    # PASO 2: Dividir y escalar los datos
    X_train, X_test, y_train, y_test, scaler = paso_2_escalar_y_dividir_datos(
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
        mejor_modelo, X_test, y_test, scaler, optimal_threshold, FEATURE_NAMES
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