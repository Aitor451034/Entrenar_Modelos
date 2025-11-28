# ==============================================================================
# MÓDULO: CatBoost para Detección de Defectos en Soldadura
# ==============================================================================
"""
Script para entrenar un modelo de CatBoost
con el objetivo de detectar puntos de soldadura defectuosos (pegados).

El proceso incluye:
1.  Carga de datos y extracción de 32 características (feature engineering).
2.  Separación de datos en entrenamiento (Train) y prueba (Test) y escalado.
3.  Definición de un pipeline que:
    a. Aplica escalado de características.
    b. Selecciona las mejores características con RFE.
    c. Entrena un modelo CatBoostClassifier con pesos de clase.
4.  Búsqueda exhaustiva de hiperparámetros (GridSearchCV) en el pipeline.
5.  Optimización del umbral de decisión (Regla de Sinergia).
6.  Evaluación final y análisis de errores en el conjunto de prueba (Test set).
7.  Guardado del pipeline completo CatBoost y el umbral.
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
from catboost import CatBoostClassifier
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
from sklearn.pipeline import Pipeline


# ==============================================================================
# 2. CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

RUTA_CSV_POR_DEFECTO = r"C:\Users\U5014554\Desktop\EntrenarModelo\DATA\Datos_Titanio25-26.csv"

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
RANDOM_STATE_SEED = 22
N_SPLITS_CV = 5
FBETA_BETA = 2
# Precisión mínima cambiada por el usuario
PRECISION_MINIMA = 0.74


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
            min_len = min(len(t_soldadura), len(valores_resistencia),len(valores_corriente), len(valores_voltaje))
            t_soldadura = t_soldadura[:min_len]
            valores_resistencia = valores_resistencia[:min_len]
            valores_voltaje = valores_voltaje[:min_len]
            valores_corriente = valores_corriente[:min_len]
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
    
    # 1. Contar duplicados
    # Se considera duplicado si TODAS las columnas (las 32 features) son idénticas
    num_duplicados = X.duplicated().sum()
    
    if num_duplicados > 0:
        print(f"⚠ ¡ALERTA! Se encontraron {num_duplicados} muestras duplicadas exactas.")
        
        # 2. (Opcional) Ver cuáles son para curiosear
        # Muestra las primeras 5 filas que son copias de otras
        print("Ejemplo de duplicados:")
        print(X[X.duplicated()].head())

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
    *** LÓGICA CENTRAL: Pipeline Completo (Scaler + Selector + CatBoost) ***
    Configura y ejecuta GridSearchCV en un pipeline completo para prevenir Data Leakage.
    
    Este pipeline encadena tres pasos:
    1. Escalado de características (StandardScaler)
    2. Selección de características (RFE con RandomForest)
    3. Clasificación (CatBoostClassifier)
    
    La búsqueda exhaustiva (GridSearchCV) optimiza hiperparámetros para maximizar F2-score.
    """
    print("Iniciando búsqueda de hiperparámetros para Pipeline Completo (Scaler + Selector + CatBoost)...")
    
    # Configurar validación cruzada estratificada para mantener proporciones de clases
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Crear métrica personalizada F2 (da más peso al Recall que a Precision)
    f2_scorer = make_scorer(fbeta_score, beta=fbeta)
    
    # --- CÁLCULO DE PESOS DE CLASE ---
    # Estos pesos compensan el desbalance de clases en CatBoost
    # Las clases minoritarias reciben mayor peso durante el entrenamiento
    n_pos = sum(y_train == 1)  # Número de muestras con defecto (clase positiva)
    n_neg = sum(y_train == 0)  # Número de muestras sin defecto (clase negativa)
    total = n_pos + n_neg

    # Pesos inversamente proporcionales al número de muestras de cada clase
    w0 = total / (2 * n_neg)  # Peso para clase mayoritaria (sin defecto)
    w1 = total / (2 * n_pos)  # Peso para clase minoritaria (con defecto)

    class_weights = [w0, w1]
    print(f"Pesos de clase calculados: w0={w0:.4f}, w1={w1:.4f}")

    # --- CONSTRUCCIÓN DEL PIPELINE ---
    # El pipeline encadena transformaciones y el modelo final
    # Esto garantiza que el escalado y selección se apliquen consistentemente
    # en entrenamiento, validación y prueba
    pipeline_cb = Pipeline([
        # Paso 1: Normalizar características al rango [-1, 1]
        ('scaler', StandardScaler()),
        
        # Paso 2: Seleccionar las mejores características usando Recursive Feature Elimination (RFE)
        # RFE entrena un RandomForest y elimina iterativamente las features menos importantes
        ('selector', RFE(
            RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
            step=0.1,  # Elimina el 10% de features en cada iteración (máxima precisión)
            verbose=0
        )),
        
        # Paso 3: Modelo clasificador CatBoost (Gradient Boosting)
        # CatBoost es robusto con features categóricas y maneja el desbalance de clases bien
         ('model', CatBoostClassifier(
            loss_function="Logloss",              # Función de pérdida (logarítmica binaria)
            eval_metric="Recall",                 # Métrica a monitorear durante entrenamiento
            depth=4,                              # Profundidad inicial de los árboles
            learning_rate=0.05,                   # Tasa de aprendizaje inicial
            bootstrap_type='Bernoulli',           # Tipo de bootstrapping (necesario para subsample)
            iterations=500,                       # Número máximo de iteraciones (árboles)
            l2_leaf_reg=8,                        # Regularización L2 (previene overfitting)
            random_seed=random_state,             # Semilla para reproducibilidad
            subsample=0.7,                        # Porcentaje de muestras para cada iteración
            od_type="Iter",                       # Tipo de detección de overfitting
            verbose=False,                        # No mostrar logs del entrenamiento
            class_weights=class_weights,          # Pesos para compensar desbalance de clases
            # task_type="GPU",    # Descomenta para usar GPU (si está disponible)
            # devices='0',        # Especifica dispositivo GPU
        ))
    ])

    # --- DEFINICIÓN DE LA GRILLA DE HIPERPARÁMETROS ---
    # GridSearchCV probará todas las combinaciones posibles de estos valores
    # Los nombres con prefijo 'model__' son parámetros del modelo CatBoost
    # Los nombres con prefijo 'selector__' son parámetros de RFE
    param_grid_cb = {
        # Parámetros del modelo CatBoost
        'model__depth': [4, 6, 8],                          # Profundidad de los árboles (mayor = más complejo)
        'model__l2_leaf_reg': [3, 7],                       # Regularización (mayor = menos overfitting)
        'model__learning_rate': [0.03, 0.06],              # Tasa de aprendizaje (menor = aprendizaje más lento pero potencialmente mejor)
        'model__subsample': [0.7, 0.85],                   # Porcentaje de muestras para entrenar cada árbol
        'model__min_data_in_leaf': [1, 5],                 # Número mínimo de muestras en hojas (previene overfitting)
        
        # Parámetro del selector de características (RFE)
        'selector__n_features_to_select': [15,20,25],    # Número final de características a seleccionar
    }
    
    # Calcular número total de combinaciones a evaluar
    total_combinaciones = np.prod([len(v) for v in param_grid_cb.values()])
    print(f"GridSearchCV (CatBoost) probará {total_combinaciones} combinaciones de hiperparámetros.")
    print("Entrenando... (Esto puede tardar varios minutos)\n")

    # --- EJECUTAR BÚSQUEDA EXHAUSTIVA (GridSearchCV) ---
    # GridSearchCV entrena y valida el pipeline con cada combinación de parámetros
    # Selecciona la combinación que maximiza la métrica F2-score en validación cruzada
    search_cv = GridSearchCV(
        estimator=pipeline_cb,
        param_grid=param_grid_cb,
        cv=skf,                      # Usar validación cruzada estratificada
        scoring=f2_scorer,            # Métrica a optimizar (F2-score)
        n_jobs=-1,                    # Usar todos los núcleos disponibles
        verbose=2,                    # Mostrar progreso
        refit=True                    # Reentranar con mejores parámetros en todo el set
    )

    # Entrenar el pipeline con la búsqueda de hiperparámetros
    search_cv.fit(X_train, y_train)
    
    # Extraer el modelo entrenado con los mejores parámetros
    mejor_modelo = search_cv.best_estimator_
    
    print("\n" + "="*70)
    print("Entrenamiento (GridSearchCV) de CatBoost completado.")
    print(f"Mejores parámetros encontrados:\n{search_cv.best_params_}")
    print(f"Mejor score F2 (en validación cruzada): {search_cv.best_score_:.4f}")
    print("="*70 + "\n")
    
    return mejor_modelo

def paso_4_evaluar_importancia_y_umbral_defecto(mejor_modelo, X_test, y_test, feature_names):
    """
    Grafica la importancia de características y la matriz de confusión
    con el umbral por defecto (0.5).
    """
    
    # --- 1. EXTRAE LOS NOMBRES DE LAS CARACTERÍSTICAS SELECCIONADAS ---
    # El selector (RFE) ha elegido un subconjunto de las 32 características originales
    # Necesitamos recuperar cuáles fueron seleccionadas para etiquetar correctamente el gráfico
    
    # Obtener el objeto selector del pipeline
    selector = mejor_modelo.named_steps['selector']
    
    # get_support() devuelve un array booleano: True si la característica fue seleccionada, False si no
    mask = selector.get_support()
    
    # Usar la máscara para filtrar los nombres originales de las 32 características
    # Solo quedan los nombres de las características que RFE eligió
    nombres_finales = np.array(feature_names)[mask]
    
    # --- 2. OBTIENE LAS IMPORTANCIAS DEL MODELO ENTRENADO ---
    # CatBoost calcula qué tan importante es cada característica para las predicciones
    # Las importancias están ordenadas según el orden de nombres_finales
    importancias = mejor_modelo.named_steps['model'].feature_importances_

    # --- 3. CREA UN DATAFRAME CON LAS IMPORTANCIAS ---
    # Organizamos los nombres y sus importancias en una tabla para análisis
    df_importancias = pd.DataFrame({
        'predictor': nombres_finales,           # Nombre de cada característica seleccionada
        'importancia': importancias              # Valor de importancia del modelo
    }).sort_values(by='importancia', ascending=True)  # Ordena de menor a mayor importancia

    # --- 4. IMPRIME REPORTE DE IMPORTANCIAS ---
    print(f"\nImportancia de las {len(df_importancias)} características seleccionadas:")
    # Ordena de MAYOR a MENOR importancia para facilitar la lectura
    print(df_importancias.sort_values(by='importancia', ascending=False))

    # --- 5. GRAFICA LAS IMPORTANCIAS EN UN GRÁFICO HORIZONTAL ---
    # Este tipo de gráfico (barh = bar horizontal) es ideal para comparar muchas variables
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crea barras horizontales con la importancia de cada característica
    ax.barh(df_importancias.predictor, df_importancias.importancia)
    
    # Reduce el tamaño de las etiquetas del eje Y para que sea legible
    plt.yticks(size=8)
    
    # Etiqueta y título del gráfico
    ax.set_xlabel('Importancia de la Característica')
    ax.set_ylabel('Variable Predictora')
    ax.set_title('Importancia de Características (CatBoost)')
    
    # Ajusta automáticamente los espacios para que no se corten las etiquetas
    plt.tight_layout()
    plt.show()

    # --- 6. OBTIENE LAS PREDICCIONES DEL MODELO EN EL CONJUNTO TEST ---
    # Usa el umbral por defecto de 0.5 (probabilidad >= 0.5 se clasifica como defecto)
    predicciones_defecto = mejor_modelo.predict(X_test)

    # --- 7. CREA LA MATRIZ DE CONFUSIÓN PARA EVALUAR EL DESEMPEÑO ---
    # La matriz de confusión muestra:
    # - Verdaderos Negativos (TN): Predijo sin defecto, era sin defecto ✓
    # - Falsos Positivos (FP): Predijo defecto, era sin defecto ✗
    # - Falsos Negativos (FN): Predijo sin defecto, era defecto ✗
    # - Verdaderos Positivos (TP): Predijo defecto, era defecto ✓
    matriz_confusion = confusion_matrix(y_test, predicciones_defecto)
    
    # --- 8. GRAFICA LA MATRIZ DE CONFUSIÓN ---
    # Usa la función auxiliar para crear un heatmap visualmente atractivo
    titulo = "Matriz de Confusión - CATBOOST (Umbral = 0.5)"
    _plot_confusion_matrix(matriz_confusion, titulo)

def paso_5_optimizar_umbral(mejor_modelo, X_train, y_train, n_splits, precision_minima, random_state):
    """
    Busca el umbral de decisión óptimo usando predicciones "Out-of-Fold" (OOF)
    Y ADEMÁS, grafica la relación entre el umbral, la precisión y el recall.
        
    El objetivo es encontrar el umbral que MAXIMIZA el Recall (sensibilidad para detectar defectos)
    mientras se mantiene la Precisión por encima de un mínimo aceptable.
        
    Parámetros:
    -----------
    mejor_modelo : Pipeline entrenado con Scaler + Selector + CatBoost
    X_train : DataFrame de características de entrenamiento
    y_train : Series con etiquetas (0 = sin defecto, 1 = con defecto)
    n_splits : número de folds para validación cruzada
    precision_minima : restricción mínima de precisión (parámetro de negocio)
    random_state : semilla para reproducibilidad
        
    Retorna:
    --------
    optimal_threshold : valor del umbral óptimo encontrado (entre 0 y 1)
    """
        
    print(f"\nOptimizando umbral para: MAX(Recall) sujeto a Precision >= {precision_minima}...")
        
    # ==== PASO 1: CONFIGURAR VALIDACIÓN CRUZADA ESTRATIFICADA ====
    # Asegura que cada fold mantenga la proporción de clases del dataset original
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ==== PASO 2: OBTENER PREDICCIONES DE VALIDACIÓN CRUZADA (OOF) ====
    # cross_val_predict genera predicciones "Out-of-Fold":
    # - Cada muestra se predice usando el modelo entrenado en los otros folds
    # - Evita usar datos de entrenamiento en las predicciones (sin overfitting)
    # - El resultado [:, 1] extrae solo las probabilidades de la clase positiva (defecto)
    print("Obteniendo predicciones de validación cruzada (OOF)...")
    y_probas_cv = cross_val_predict(
        mejor_modelo,
        X_train,
        y_train,
        cv=skf,
        method='predict_proba',  # Retorna probabilidades [0-1] en lugar de clases [0 o 1]
        n_jobs=-1                 # Usar todos los núcleos CPU disponibles
    )[:, 1]  # Extraer probabilidades de la clase 1 (defecto)

    # ==== PASO 3: PREPARAR LISTAS PARA ALMACENAR RESULTADOS ====
    # Calcularemos Precision y Recall para cada umbral posible
    lista_umbrales = np.linspace(0.01, 0.99, 1000)  # 1000 umbrales entre 0.01 y 0.99
    lista_precision = []  # Almacena precisión para cada umbral
    lista_recall = []     # Almacena recall para cada umbral

    # ==== PASO 4: VARIABLES PARA RASTREAR EL MEJOR UMBRAL ====
    best_recall = -1                    # Mejor recall encontrado hasta ahora
    optimal_threshold = 0.5              # Umbral por defecto (fallback)
    best_precision_at_best_recall = 0   # Precisión en el mejor umbral

    # ==== PASO 5: EVALUAR CADA UMBRAL CANDIDATO ====
    print("Calculando métricas (Precision y Recall) para cada umbral...")
    for thresh in lista_umbrales:
        # Convertir probabilidades a predicciones binarias usando el umbral actual
        # Si probabilidad >= umbral → predecir defecto (1), sino → sin defecto (0)
        y_pred_thresh = np.where(y_probas_cv >= thresh, 1, 0)

        # Calcular Precision: ¿De los casos que predijo como defecto, cuántos eran realmente defecto?
        prec = precision_score(y_train, y_pred_thresh, zero_division=0)
            
        # Calcular Recall: ¿De los defectos reales, cuántos logró detectar?
        rec = recall_score(y_train, y_pred_thresh, zero_division=0)

        # Guardar valores para graficar después
        lista_precision.append(prec)
        lista_recall.append(rec)

        # ==== LÓGICA DE OPTIMIZACIÓN: Buscar el mejor umbral ====
        # Restricción: La precisión debe cumplir con el mínimo requerido
        if prec >= precision_minima:
            # Si cumple la restricción, buscar el que tenga mayor Recall (máxima detección de defectos)
            if rec > best_recall:
                # Nuevo mejor recall encontrado
                best_recall = rec
                optimal_threshold = thresh
                best_precision_at_best_recall = prec
            elif rec == best_recall:
                # Si el recall es idéntico, elegir el umbral más bajo (menos restrictivo)
                optimal_threshold = min(optimal_threshold, thresh)

    # ==== PASO 6: VALIDAR QUE SE ENCONTRÓ UN UMBRAL VÁLIDO ====
    if best_recall == -1:
        # Si no se encontró ningún umbral que cumpla la restricción de precisión
        print(f"¡ADVERTENCIA! No se encontró NINGÚN umbral que cumpla 'Precision >= {precision_minima}'.")
        print("El modelo no puede satisfacer este requisito. Usando 0.5 por defecto.")
        optimal_threshold = 0.5
    else:
        # Éxito: Se encontró un umbral que satisface todas las restricciones
        print("¡Éxito! Se encontró un umbral que cumple los requisitos.")
        print(f"   → Umbral óptimo: {optimal_threshold:.4f}")
        print(f"   → Recall resultante (en CV): {best_recall:.4f} (máximo posible)")
        print(f"   → Precision resultante (en CV): {best_precision_at_best_recall:.4f} (cumple mínimo)")

    # ==== PASO 7: GENERAR GRÁFICO DE PRECISION-RECALL vs UMBRAL ====
    # Este gráfico ayuda a visualizar cómo cambian las métricas al variar el umbral
    print("Generando gráfico Precision-Recall vs. Umbral...")
    plt.figure(figsize=(12, 7))
        
    # Línea de Precisión: cómo cambia la precisión con cada umbral
    # A mayor umbral → mayor precisión (menos falsos positivos)
    plt.plot(lista_umbrales, lista_precision, label='Precision', color='blue', linewidth=2)
        
    # Línea de Recall: cómo cambia el recall con cada umbral
    # A mayor umbral → menor recall (menos detecciones de defectos)
    plt.plot(lista_umbrales, lista_recall, label='Recall', color='green', linewidth=2)
        
    # Línea vertical en rojo: marca el umbral óptimo encontrado
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Umbral Óptimo ({optimal_threshold:.4f})')
        
    # Línea horizontal gris: marca la restricción de precisión mínima
    # La región válida es ARRIBA de esta línea
    plt.axhline(y=precision_minima, color='gray', linestyle=':', linewidth=2,
                label=f'Precisión Mínima ({precision_minima})')
        
    # ==== CONFIGURACIÓN DEL GRÁFICO ====
    plt.title('Precision y Recall vs. Umbral de Decisión\n(Validación Cruzada OOF)', fontsize=16, fontweight='bold')
    plt.xlabel('Umbral de Decisión (Threshold)', fontsize=12)
    plt.ylabel('Puntuación (0.0 a 1.0)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
        
    # Añadir marcas cada 5% para facilitar la lectura
    ticks = np.arange(0, 1.05, 0.05)
    plt.xticks(ticks)
    plt.yticks(ticks)
        
    # Establecer límites visibles con pequeño margen superior
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.show()

    # ==== RETORNAR EL UMBRAL ÓPTIMO ====
    return optimal_threshold

def paso_6_evaluacion_final_y_guardado(mejor_modelo, X_test, y_test, scaler, optimal_threshold, feature_names):
    """
    Evaluación final en el Test set: Reporte, Matriz de Confusión,
    Curva ROC, Análisis de Errores y Guardado del modelo.
    
    Este paso es crucial para entender el desempeño real del modelo en datos no vistos.
    Se ejecuta DESPUÉS de haber optimizado el umbral de decisión.
    
    Parámetros:
    -----------
    mejor_modelo : Pipeline entrenado (Scaler + Selector + CatBoost)
    X_test : DataFrame con características del conjunto de prueba
    y_test : Series con etiquetas reales del conjunto de prueba (0 o 1)
    scaler : Objeto StandardScaler (para referencia, aunque ya está en el pipeline)
    optimal_threshold : Umbral de decisión optimizado en el paso anterior
    feature_names : Lista con nombres de las 32 características originales
    """
    print("\n--- Evaluación Final en Conjunto de Prueba (Test Set) ---")
    
    # ==== PASO 1: GENERAR PREDICCIONES EN EL TEST SET ====
    # El pipeline aplica automáticamente: Scaler → Selector → Modelo
    # predict_proba retorna dos columnas [P(clase 0), P(clase 1)]
    # Extraemos [:, 1] para obtener solo la probabilidad de defecto (clase 1)
    predicciones_test_proba = mejor_modelo.predict_proba(X_test)[:, 1]
    
    # Convertir probabilidades a predicciones binarias usando el umbral optimizado
    # Si P(defecto) >= optimal_threshold → predecir 1 (defecto)
    # Si P(defecto) < optimal_threshold → predecir 0 (sin defecto)
    predicciones_test_binarias = np.where(predicciones_test_proba >= optimal_threshold, 1, 0)

    # ==== PASO 2: REPORTE DE CLASIFICACIÓN ====
    # Este reporte muestra:
    # - Precision: De los casos predichos como defecto, ¿cuántos eran correctos?
    # - Recall: De los defectos reales, ¿cuántos logró detectar?
    # - F1-Score: Media armónica entre Precision y Recall
    # - Support: Número de muestras en cada clase en el conjunto test
    print("\nReporte de Clasificación (Test Set):")
    target_names = ['0: Sin Defecto', '1: Con Defecto (Pegado)']
    print(classification_report(y_test, predicciones_test_binarias, target_names=target_names))

    # ==== PASO 3: MATRIZ DE CONFUSIÓN CON UMBRAL ÓPTIMO ====
    # La matriz de confusión muestra la distribución de aciertos y errores:
    # [[TN, FP],
    #  [FN, TP]]
    # Donde:
    # TN = Verdaderos Negativos (predijo sin defecto, era sin defecto) ✓
    # FP = Falsos Positivos (predijo defecto, era sin defecto) ✗
    # FN = Falsos Negativos (predijo sin defecto, era defecto) ✗
    # TP = Verdaderos Positivos (predijo defecto, era defecto) ✓
    matriz_confusion_opt = confusion_matrix(y_test, predicciones_test_binarias)
    
    # Graficar la matriz de confusión con el umbral óptimo en el título
    titulo = f"Matriz de Confusión - CATBOOST (Umbral Óptimo = {optimal_threshold:.4f})"
    _plot_confusion_matrix(matriz_confusion_opt, titulo)

    # ==== PASO 4: CURVA ROC (RECEIVER OPERATING CHARACTERISTIC) ====
    # La curva ROC muestra cómo varían los verdaderos positivos vs falsos positivos
    # con TODOS los umbrales posibles (no solo el óptimo).
    # 
    # IMPORTANTE: Usar probabilidades (predict_proba), NO las predicciones binarias.
    # Las predicciones binarias solo dan dos puntos: (0,0) y (1,1).
    # Las probabilidades generan la curva completa.
    fpr, tpr, _ = metrics.roc_curve(y_test, predicciones_test_proba)
    auc_score = metrics.roc_auc_score(y_test, predicciones_test_proba)
    
    # AUC (Area Under Curve) = probabilidad de que el modelo rankee un defecto
    # más alto que un no-defecto. Rango: 0 a 1. Mayor es mejor.
    # 0.5 = modelo aleatorio
    # 1.0 = modelo perfecto
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"CATBOOST (AUC = {auc_score:.4f})")
    
    # Línea diagonal = desempeño aleatorio (baseline)
    plt.plot([0, 1], [0, 1], 'k--', label="Clasificador Aleatorio (AUC = 0.5)")
    
    plt.xlabel('Tasa de Falsos Positivos (FPR) - Falsas Alarmas')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR) - Detecciones Correctas')
    plt.title('Curva ROC (Test Set)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # ==========================================================================
    # PASO 5: ANÁLISIS DETALLADO DE ERRORES (Falsos Negativos y Falsos Positivos)
    # ==========================================================================
    # Este análisis es CRÍTICO para entender dónde falla el modelo
    # Nos permite identificar patrones en los errores
    print("\n--- INICIANDO ANÁLISIS DE ERRORES EN EL TEST SET ---")

    # Crear DataFrame de análisis que preserva los índices originales del test set
    # Esto es importante para rastrear qué muestras específicas fallaron
    df_analisis = pd.DataFrame(y_test)

    # Añadir columnas con probabilidades y predicciones binarias
    # Los índices se alinean automáticamente porque están en el mismo orden
    df_analisis['Probabilidad_Defecto'] = predicciones_test_proba
    df_analisis['Prediccion_Binaria'] = predicciones_test_binarias
    
    # --- IDENTIFICAR FALSOS NEGATIVOS (FN) ---
    # Definición: Real = Defecto (1), Predicción = Sin Defecto (0)
    # Problema: El modelo NO detectó un defecto que SÍ existía
    # Consecuencia: Riesgo de producción - se envía un producto defectuoso
    # Métrica clave: Recall = TP / (TP + FN). El objetivo es minimizar FN.
    
    # Crear máscara booleana para identificar filas que son FN
    condicion_FN = (df_analisis['Etiqueta_Defecto'] == 1) & (df_analisis['Prediccion_Binaria'] == 0)
    falsos_negativos = df_analisis[condicion_FN]
    
    print(f"\n[INFORME] Se han encontrado {len(falsos_negativos)} Falsos Negativos (Defectos NO detectados):")
    print("Estos casos son CRÍTICOS: El modelo falló en detectar defectos reales.")
    if len(falsos_negativos) > 0:
        # Mostrar índices y probabilidades de los falsos negativos
        print(falsos_negativos[['Etiqueta_Defecto', 'Prediccion_Binaria', 'Probabilidad_Defecto']].to_string())
        print(f"→ Probabilidades promedio en FN: {falsos_negativos['Probabilidad_Defecto'].mean():.4f}")
        print("→ Estos casos tuvieron baja confianza de defecto pero aún así eran defectos.")
    else:
        print("✓ ¡Excelente! No hay falsos negativos en el test set.")

    # --- IDENTIFICAR FALSOS POSITIVOS (FP) ---
    # Definición: Real = Sin Defecto (0), Predicción = Defecto (1)
    # Problema: El modelo PREDIJO defecto cuando NO había defecto (falsa alarma)
    # Consecuencia: Costo de reproceso innecesario, pero no es defecto final
    # Métrica clave: Precision = TP / (TP + FP). El objetivo es mantenerla alta.
    
    # Crear máscara booleana para identificar filas que son FP
    condicion_FP = (df_analisis['Etiqueta_Defecto'] == 0) & (df_analisis['Prediccion_Binaria'] == 1)
    falsos_positivos = df_analisis[condicion_FP]
    
    print(f"\n[INFORME] Se han encontrado {len(falsos_positivos)} Falsos Positivos (Falsas Alarmas):")
    print("Estos casos son ACEPTABLES pero costosos: Se rechazan productos buenos.")
    if len(falsos_positivos) > 0:
        # Mostrar índices y probabilidades de los falsos positivos
        print(falsos_positivos[['Etiqueta_Defecto', 'Prediccion_Binaria', 'Probabilidad_Defecto']].to_string())
        print(f"→ Probabilidades promedio en FP: {falsos_positivos['Probabilidad_Defecto'].mean():.4f}")
        print("→ El modelo fue demasiado agresivo prediciendo defectos aquí.")
    else:
        print("✓ ¡Excelente! No hay falsos positivos en el test set.")

    # ==== PASO 6: GUARDAR ARTEFACTOS DEL MODELO ====
    # Se guarda:
    # 1. Pipeline completo: Contiene todo (Scaler, Selector, Modelo)
    # 2. Umbral optimizado: Se necesita para hacer nuevas predicciones
    # 3. Nombres de features: Para documentación y trazabilidad
    
    print("\nGuardando pipeline COMPLETO (Scaler+SMOTE+Selector+Modelo) y umbral...")
    
    # Diccionario con todos los artefactos necesarios para usar el modelo después
    artefactos_modelo = {
        "pipeline_completo": mejor_modelo,  # El modelo entrenado con todos sus pasos
        "umbral": optimal_threshold,         # Umbral de decisión optimizado
        "feature_names_originales": feature_names  # Para saber qué features espera el modelo
    }
    
    # Nombre del archivo que se guardará en disco
    nombre_archivo = 'modelo_con_umbral_PEGADOS_PipelineCompleto.pkl'
    
    # Guardar usando pickle (serialización de Python)
    with open(nombre_archivo, 'wb') as f:
        pickle.dump(artefactos_modelo, f)

    print(f"✓ ¡Proceso completado! Modelo guardado con umbral = {optimal_threshold:.4f}")
    print(f"  Archivo: {nombre_archivo}")
    print("  (Este archivo contiene TODO lo necesario para hacer nuevas predicciones)")

def _plot_confusion_matrix(cm, title):
    """
    Función auxiliar para graficar una matriz de confusión de manera visual.
    
    Parámetros:
    -----------
    cm : array 2x2 con la matriz de confusión
         [[TN, FP],
          [FN, TP]]
    title : string con el título del gráfico
    """
    fig, ax = plt.subplots()
    
    # Crear heatmap (mapa de calor) con la matriz de confusión
    # Los valores más altos se ven en colores más oscuros
    sns.heatmap(
        pd.DataFrame(cm),
        annot=True,          # Mostrar valores en las celdas
        cmap="YlGnBu",       # Paleta de colores (amarillo → azul)
        fmt='g',             # Formato de los números
        cbar=False,          # No mostrar barra de color a la derecha
        xticklabels=["Predicho Sin Defecto", "Predicho Pegado"],
        yticklabels=["Real Sin Defecto", "Real Pegado"]
    )
    
    # Configuración de etiquetas y título
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
    Función principal que orquesta todo el pipeline de Machine Learning.
    
    Este es el flujo completo del proyecto:
    1. Cargar datos crudos y calcular 32 características
    2. Dividir en Train/Test y escalar características
    3. Entrenar pipeline con GridSearchCV (optimización de hiperparámetros)
    4. Evaluar importancia de características y desempeño inicial
    5. Optimizar el umbral de decisión (Precision vs Recall)
    6. Evaluación final en Test set y guardado del modelo
    7. Análisis de sesgo-varianza (Bias-Variance Trade-off)
    """
    
    # ==========================================================================
    # PASO 1: CARGAR Y PROCESAR DATOS CRUDOS
    # ==========================================================================
    # Este paso:
    # - Abre un diálogo para seleccionar el archivo CSV con datos de soldadura
    # - Limpia y preprocesa los datos (extrae columnas relevantes, elimina ruido)
    # - Calcula las 32 características de ingeniería (feature engineering)
    # - Retorna dos matrices: X (características) e y (etiquetas de defecto)
    # 
    # Si el usuario cancela el diálogo o hay error, devuelve None
    print("\n" + "="*70)
    print("PASO 1: CARGANDO Y PREPARANDO DATOS")
    print("="*70)
    X, y = paso_1_cargar_y_preparar_datos(FEATURE_NAMES)
    if X is None:
        print("ERROR: No se pudieron cargar los datos. Terminando ejecución.")
        return
    
    # --- NUEVO: DIAGNÓSTICO DE ENERGÍA ---
    # Esto te dirá si el problema es "demasiado fácil" físicamente
    graficar_distribucion_energia(X, y)
    # -------------------------------------
    
    # ==========================================================================
    # PASO 2: DIVIDIR DATOS EN TRAIN/TEST Y ESCALAR
    # ==========================================================================
    # Este paso:
    # - Divide los datos en 60% entrenamiento y 40% prueba (TEST_SIZE_RATIO = 0.4)
    # - Usa StratifiedKFold para mantener la proporción de clases en ambos sets
    # - Los datos se escalarán DENTRO del pipeline en el Paso 3 (evita Data Leakage)
    #
    # Data Leakage: Error común de usar estadísticas del FULL dataset para escalar.
    # Solución correcta: Escalar solo con estadísticas del Train set.
    print("\n" + "="*70)
    print("PASO 2: DIVIDIENDO DATOS EN ENTRENAMIENTO (60%) Y PRUEBA (40%)")
    print("="*70)
    X_train, X_test, y_train, y_test, scaler = paso_2_escalar_y_dividir_datos(
        X, y, TEST_SIZE_RATIO, RANDOM_STATE_SEED
    )
    print(f"✓ Entrenamiento: {len(X_train)} muestras")
    print(f"✓ Prueba: {len(X_test)} muestras")

    # ==========================================================================
    # PASO 3: ENTRENAR MODELO CON BÚSQUEDA DE HIPERPARÁMETROS
    # ==========================================================================
    # Este es el paso MÁS IMPORTANTE y LARGO del pipeline.
    # 
    # Qué hace:
    # 1. Crea un Pipeline que encadena tres transformaciones:
    #    a) StandardScaler: Normaliza características al rango [-1, 1]
    #    b) RFE: Selecciona las mejores ~20 características (elimina ruido)
    #    c) CatBoost: Modelo de Gradient Boosting (árbol mejorado iterativamente)
    #
    # 2. Ejecuta GridSearchCV que prueba 3×2×2×2×2×3 = 288 combinaciones de:
    #    - Profundidad del árbol: [4, 6, 8]
    #    - Regularización L2: [3, 7]
    #    - Tasa de aprendizaje: [0.03, 0.06]
    #    - Subsample (% muestras por árbol): [0.7, 0.85]
    #    - Mínimo de datos en hojas: [1, 5]
    #    - Número de features a seleccionar: [15, 20, 25]
    #
    # 3. Usa validación cruzada estratificada (5 folds) para evaluar cada combo
    #
    # 4. Selecciona la combinación que MAXIMIZA el F2-score
    #    (F2 da más peso al Recall que a Precision, importante para detectar defectos)
    #
    # ⚠ ADVERTENCIA: Este paso puede tardar 30+ minutos (depende del hardware)
    print("\n" + "="*70)
    print("PASO 3: ENTRENANDO MODELO CON BÚSQUEDA DE HIPERPARÁMETROS (GridSearchCV)")
    print("="*70)
    print("⏳ ADVERTENCIA: Este paso puede tardar varios minutos...")
    print("   (Probando 288 combinaciones de hiperparámetros con validación cruzada)")
    mejor_modelo = paso_3_entrenar_modelo(
        X_train, y_train, 
        N_SPLITS_CV, FBETA_BETA, RANDOM_STATE_SEED
    )
    print("✓ Entrenamiento completado. Mejor modelo guardado.")

    # ==========================================================================
    # PASO 4: EVALUAR IMPORTANCIA Y DESEMPEÑO INICIAL
    # ==========================================================================
    # Este paso genera dos visualizaciones importantes:
    #
    # 1. GRÁFICO DE IMPORTANCIA DE CARACTERÍSTICAS:
    #    - Muestra cuáles de las 32 features originales son MÁS útiles
    #    - Las features con mayor importancia tienen más poder predictivo
    #    - Uso: Entender qué patrones el modelo detecta en los datos
    #
    # 2. MATRIZ DE CONFUSIÓN (con umbral por defecto = 0.5):
    #    - Muestra aciertos y errores del modelo
    #    - TN (arriba-izq): Predijo sin defecto, ERA sin defecto ✓
    #    - FP (arriba-der): Predijo defecto, ERA sin defecto ✗ (falsa alarma)
    #    - FN (abajo-izq): Predijo sin defecto, ERA defecto ✗ (defecto no detectado)
    #    - TP (abajo-der): Predijo defecto, ERA defecto ✓
    #
    # Nota: El umbral 0.5 probablemente NO es óptimo. Se optimiza en el Paso 5.
    print("\n" + "="*70)
    print("PASO 4: EVALUANDO IMPORTANCIA DE CARACTERÍSTICAS Y DESEMPEÑO INICIAL")
    print("="*70)
    paso_4_evaluar_importancia_y_umbral_defecto(
        mejor_modelo, X_test, y_test, FEATURE_NAMES
    )
    print("✓ Gráficos generados: Importancia y Matriz de Confusión (umbral=0.5)")

    # ==========================================================================
    # PASO 5: OPTIMIZAR EL UMBRAL DE DECISIÓN
    # ==========================================================================
    # Este paso es CRÍTICO para el desempeño en producción.
    #
    # ¿Qué es el umbral de decisión?
    # - El modelo CatBoost genera probabilidades entre 0 y 1 (confianza de defecto)
    # - Con umbral = 0.5: Si probabilidad >= 0.5 → predecir defecto, si no → sin defecto
    # - Un umbral diferente (ej. 0.3) es más "agresivo" (detecta más defectos)
    # - Un umbral más alto (ej. 0.7) es más "conservador" (menos falsas alarmas)
    #
    # ¿Por qué optimizar?
    # - Diferentes aplicaciones necesitan diferentes balances
    # - En soldadura, detectar defectos (Recall) es MÁS importante que estar "seguro" (Precision)
    # - Pero necesitamos garantizar una precisión MÍNIMA (PRECISION_MINIMA = 0.81)
    #
    # Algoritmo:
    # 1. Genera predicciones "Out-of-Fold" con validación cruzada (sin overfitting)
    # 2. Prueba 1000 umbrales diferentes (0.01 a 0.99)
    # 3. Para cada umbral: calcula Precision y Recall
    # 4. Elige el umbral que MAXIMIZA Recall PERO mantiene Precision >= 0.81
    # 5. Genera gráfico mostrando cómo varían Precision y Recall con el umbral
    #
    # Output: Un umbral óptimo (ej. 0.35) que será usado en el Paso 6
    print("\n" + "="*70)
    print("PASO 5: OPTIMIZANDO UMBRAL DE DECISIÓN")
    print("="*70)
    print(f"Criterio de optimización:")
    print(f"  - MAXIMIZAR: Recall (detección de defectos)")
    print(f"  - RESTRICCIÓN: Precision >= {PRECISION_MINIMA}")
    optimal_threshold = paso_5_optimizar_umbral(
        mejor_modelo, X_train, y_train, 
        N_SPLITS_CV, PRECISION_MINIMA, RANDOM_STATE_SEED
    )
    print(f"✓ Umbral óptimo encontrado: {optimal_threshold:.4f}")

    # ==========================================================================
    # PASO 6: EVALUACIÓN FINAL Y GUARDADO DEL MODELO
    # ==========================================================================
    # Este paso es la EVALUACIÓN DEFINITIVA del modelo en datos nuevos (Test set).
    # 
    # Qué hace:
    # 1. Genera predicciones en el Test set (40% de datos no usados en entrenamiento)
    # 2. Usa el umbral optimizado del Paso 5 (no el 0.5 por defecto)
    # 3. Imprime REPORTE DE CLASIFICACIÓN:
    #    - Precision: De los defectos predichos, ¿cuántos eran reales?
    #    - Recall: De los defectos reales, ¿cuántos detectó?
    #    - F1-Score: Media armónica entre Precision y Recall
    #    - Support: Número de muestras de cada clase en Test
    #
    # 4. Genera MATRIZ DE CONFUSIÓN con umbral óptimo
    #
    # 5. Genera CURVA ROC:
    #    - Muestra desempeño del modelo para TODOS los umbrales posibles
    #    - AUC (Area Under Curve): Métrica de 0 a 1. Mayor = mejor.
    #    - 0.5 = modelo aleatorio, 1.0 = modelo perfecto
    #
    # 6. ANÁLISIS DETALLADO DE ERRORES:
    #    - Identifica FALSOS NEGATIVOS: Defectos no detectados (CRÍTICO)
    #    - Identifica FALSOS POSITIVOS: Falsas alarmas (costoso pero seguro)
    #    - Muestra probabilidades de estos casos problemáticos
    #
    # 7. GUARDA EL MODELO EN DISCO:
    #    - Archivo: 'modelo_con_umbral_PEGADOS_PipelineCompleto.pkl'
    #    - Contiene: Pipeline completo + umbral óptimo + nombres de features
    #    - Usar después: Cargar este archivo para hacer predicciones en nuevos datos
    print("\n" + "="*70)
    print("PASO 6: EVALUACIÓN FINAL EN TEST SET Y GUARDADO DEL MODELO")
    print("="*70)
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
    
    # ==========================================================================
    # FIN DEL PIPELINE
    # ==========================================================================
    print("\n" + "="*70)
    print("✓ ¡PIPELINE COMPLETADO EXITOSAMENTE!")
    print("="*70)
    print("\nResumen:")
    print(f"  1. Se cargaron y procesaron datos de soldadura")
    print(f"  2. Se dividieron en Train ({len(X_train)} muestras) y Test ({len(X_test)} muestras)")
    print(f"  3. Se entrenó pipeline con {N_SPLITS_CV}-fold Cross-Validation")
    print(f"  4. Se optimizó umbral de decisión: {optimal_threshold:.4f}")
    print(f"  5. Se guardó modelo completo en 'modelo_con_umbral_PEGADOS_PipelineCompleto.pkl'")
    print(f"\n✓ El modelo está listo para usar en producción.")


if __name__ == "__main__":
    main()