"""
Script para entrenar un modelo de Random Forest
con el objetivo de detectar puntos de soldadura defectuosos (pegados).

El proceso incluye:
1.  Carga de datos de un CSV.
2.  Extracción y cálculo de 32 características (feature engineering) 
    a partir de las curvas de resistencia dinámica.
3.  Separación de datos en entrenamiento (Train) y prueba (Test).
4.  Escalado de características (StandardScaler).
5.  Entrenamiento del modelo usando RandomizedSearchCV para encontrar 
    los hiperparámetros óptimos, optimizando para F2-Score.
6.  Análisis de importancia de características.
7.  Optimización del umbral de decisión para maximizar el Recall 
    manteniendo una Precisión mínima.
8.  Evaluación final del modelo en el conjunto de prueba (Test set).
9.  Guardado del modelo, el escalador y el umbral óptimo en un archivo .pkl.
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
from scipy.stats import skew, kurtosis, randint

# --- Componentes de Scikit-learn ---
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc, fbeta_score, make_scorer, classification_report, confusion_matrix,
    precision_score, recall_score, roc_curve, roc_auc_score
)
from sklearn import metrics

# ==============================================================================
# 2. CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

# Ruta por defecto al archivo de datos.
# Si no se encuentra, se abrirá un diálogo para seleccionarlo.
RUTA_CSV_POR_DEFECTO = r"C:\Users\U5014554\Desktop\EntrenarModelo\DATA\Inputs_modelo_pegado_con_datos3.csv"

# Nombres de las 32 características que se generarán.
# Es crucial para la legibilidad y el guardado del modelo.
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

# Configuración del modelo y entrenamiento
TEST_SIZE_RATIO = 0.4       # Proporción de datos para el conjunto de prueba
RANDOM_STATE_SEED = 42      # Semilla para reproducibilidad
N_SPLITS_CV = 5             # Número de folds para la validación cruzada
FBETA_BETA = 2              # Usar F2-score (más peso a Recall)
N_ITER_RANDOM_SEARCH = 75   # Número de iteraciones para RandomizedSearchCV
PRECISION_MINIMA = 0.70     # Requisito para la optimización del umbral


# ==============================================================================
# 3. FUNCIONES DE CARGA Y EXTRACCIÓN DE CARACTERÍSTICAS (Feature Engineering)
# ==============================================================================

def leer_archivo(ruta_csv_defecto):
    """
    Lee un archivo CSV con datos de soldadura.
    
    Si la ruta por defecto no se encuentra, abre un diálogo para
    que el usuario seleccione el archivo manualmente.

    Args:
        ruta_csv_defecto (str): La ruta del archivo a intentar leer primero.

    Returns:
        pd.DataFrame or None: DataFrame con los datos, o None si la operación se cancela.
    """
    print("Abriendo archivo ...")
    try:
        df = pd.read_csv(ruta_csv_defecto, encoding="utf-8", sep=";", on_bad_lines="skip", header=None, quotechar='"', decimal=",", skiprows=3)
        print("¡Archivo CSV leído correctamente desde la ruta por defecto!")
        return df
    except FileNotFoundError:
        print("No se ha encontrado el archivo en la ruta por defecto. Abriendo diálogo...")
        root = tk.Tk()
        root.withdraw()  # Ocultar la ventana principal de tkinter
        ruta_csv_manual = filedialog.askopenfilename(
            title="Seleccionar archivo que contiene los datos",
            filetypes=[("Archivos de CSV", "*.csv")]
        )
        if not ruta_csv_manual:
            print("Operación cancelada por el usuario.")
            return None
        try:
            df = pd.read_csv(ruta_csv_manual, encoding="utf-8", sep=";", on_bad_lines="skip", header=None, quotechar='"', decimal=",")
            print("¡Archivo CSV leído correctamente desde la ruta seleccionada!")
            return df
        except Exception as e:
            print(f"Se produjo un error al leer el archivo seleccionado: {e}")
            return None
    except Exception as e:
        print(f"Se produjo un error inesperado al leer el archivo: {e}")
        return None

def calcular_pendiente(resistencias, tiempos):
    """
    Calcula la pendiente (tasa de cambio) entre valores consecutivos 
    de resistencia y tiempo.

    Args:
        resistencias (list): Secuencia de valores de resistencia.
        tiempos (list): Secuencia de valores de tiempo correspondientes.

    Returns:
        list: Secuencia de valores de pendiente.
    """
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
    """
    Calcula la primera, segunda y tercera derivada de la curva resistencia-tiempo.
    Usa np.gradient para una estimación numérica.

    Args:
        resistencias (list): Secuencia de valores de resistencia.
        tiempos (list): Secuencia de valores de tiempo correspondientes.

    Returns:
        tuple: (primera_derivada, segunda_derivada, tercera_derivada)
               como arrays de numpy.
    """
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
    """
    Toma el DataFrame crudo y aplica la limpieza inicial:
    selección de columnas, renombrado y conversión de tipos.
    """
    # Selección de columnas de interés
    new_df = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]]
    # Eliminar las últimas 2 filas (posiblemente metadatos)
    new_df = new_df.iloc[:-2]
    # Renombrar columnas
    new_df.columns = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "KAI2", "Ts2", "Fuerza", "Etiqueta datos"]
    
    # Conversión de tipos numéricos
    for col in ["KAI2", "Ts2", "Fuerza"]:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    
    float_cols = new_df.select_dtypes(include='float64').columns
    new_df = new_df.round({col: 4 for col in float_cols})
    
    new_df.index = range(1, len(new_df) + 1)
    
    # Corrección de comas decimales en el DataFrame original (puede ser innecesario
    # si las columnas de series temporales ya están en `new_df` y se manejan bien)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except:
                pass
                
    return new_df

def extraer_features_fila_por_fila(new_df):
    """
    Itera sobre cada fila (punto de soldadura) del DataFrame preprocesado
    y calcula el vector de 32 características.
    
    Args:
        new_df (pd.DataFrame): DataFrame con los datos limpios.
        
    Returns:
        tuple: (X_calculado, y_calculado) como arrays de numpy.
    """
    X_calculado = []
    y_calculado = []

    print(f"Iniciando cálculo de features para {len(new_df)} puntos de soldadura...")

    # Bucle principal de Feature Engineering
    for i in new_df.index:
        # --- 1. Leer series temporales (Voltaje, Corriente) ---
        datos_voltaje = new_df.loc[i, "Voltajes inst."]
        datos_corriente = new_df.loc[i, "Corrientes inst."]

        if pd.isna(datos_voltaje) or pd.isna(datos_corriente):
            print(f"Advertencia: Datos nulos en fila {i}. Saltando.")
            continue
            
        valores_voltaje = [float(v) for v in datos_voltaje.split(';') if v.strip()]
        valores_voltaje = [round(v, 0) for v in valores_voltaje]

        # Evitar división por cero en R=V/I
        valores_corriente = [0.001 if float(v) == 0 else float(v) for v in datos_corriente.split(';') if v.strip()]
        valores_corriente = [round(v, 0) for v in valores_corriente]

        # --- 2. Calcular Resistencia y Tiempo ---
        valores_resistencia = [v / c if c != 0 else 0 for v, c in zip(valores_voltaje, valores_corriente)]
        valores_resistencia = [round(r, 2) for r in valores_resistencia]
        valores_resistencia.append(0)  # Añadir 0 al final

        ns = int(new_df.loc[i, "Ns"])
        ts2 = int(new_df.loc[i, "Ts2"])
        t_soldadura = (np.linspace(0, ts2, ns + 1)).tolist()

        # Parche de seguridad por si las longitudes no coinciden
        if len(t_soldadura) != len(valores_resistencia):
            min_len = min(len(t_soldadura), len(valores_resistencia))
            t_soldadura = t_soldadura[:min_len]
            valores_resistencia = valores_resistencia[:min_len]
            valores_voltaje = valores_voltaje[:min_len] # Asegurar consistencia

        if not t_soldadura:
            print(f"Advertencia: Fila {i} sin datos de series temporales. Saltando.")
            continue

        # --- 3. Puntos clave de la curva R(t) ---
        resistencia_max = max(valores_resistencia)
        I_R_max = np.argmax(valores_resistencia)
        t_R_max = int(t_soldadura[I_R_max])
        
        r0 = valores_resistencia[0]
        t0 = t_soldadura[0]
        
        r_e = valores_resistencia[-2] # Última R antes del 0
        t_e = t_soldadura[-2]
        
        resistencia_min = min(valores_resistencia[:-1]) # R min (sin el 0 final)
        t_min = np.argmin(valores_resistencia[:-1])
        t_soldadura_min = t_soldadura[t_min]
        
        # --- 4. Parámetros escalares ---
        kAI2 = new_df.loc[i, "KAI2"]
        f = new_df.loc[i, "Fuerza"]
        
        # --- 5. Cálculo de Features (basado en el código original) ---
        
        # Pendiente total (no usada en las features, pero calculada)
        # r_final = valores_resistencia[-1]
        # t_final = t_soldadura[-1]
        # pendiente_total = np.nan_to_num((((r_final - r0) / (t_final) * 100)), nan=0)

        # Q (Calor)
        q = np.nan_to_num(((((kAI2 * 1000.0) ** 2) * (ts2 / 1000.0)) / (f * 10.0)), nan=0)
        
        # Áreas
        area_bajo_curva = np.nan_to_num(np.trapz(valores_resistencia, t_soldadura), nan=0)
        resistencia_ultima = valores_resistencia[-2]

        # Pendiente k4 (Pico -> Fin)
        try:
            delta_t_k4 = t_e - t_R_max
            k4 = 0 if delta_t_k4 == 0 else ((r_e - resistencia_max) / delta_t_k4) * 100
        except ZeroDivisionError:
            k4 = 0
        k4 = np.nan_to_num(k4, nan=0)
        
        # Pendiente k3 (Inicio -> Pico)
        try:
            delta_t_k3 = t_R_max - t0
            k3 = 0 if delta_t_k3 == 0 else ((resistencia_max - r0) / delta_t_k3) * 100
        except ZeroDivisionError:
            k3 = 0
        k3 = np.nan_to_num(k3, nan=0)

        # Estadísticas básicas R(t)
        desv = np.nan_to_num(np.std(valores_resistencia), nan=0)
        rms = np.nan_to_num(np.sqrt(np.mean(np.square(valores_resistencia))), nan=0)
        
        # Rangos
        rango_tiempo_max_min = np.nan_to_num(t_R_max - t_soldadura_min, nan=0)
        rango_rmax_rmin = np.nan_to_num(resistencia_max - resistencia_min, nan=0)
        
        # Pendiente V(t)
        voltaje_max = max(valores_voltaje)
        t_max_v = np.argmax(valores_voltaje)
        t_voltaje_max = t_soldadura[t_max_v]
        voltaje_final = valores_voltaje[-2]
        t_voltaje_final = t_soldadura[-2]
        
        try:
            delta_t_v = t_voltaje_max - t_voltaje_final
            pendiente_V = 0 if delta_t_v == 0 else ((voltaje_max - voltaje_final) / delta_t_v)
        except ZeroDivisionError:
            pendiente_V = 0
        pendiente_V = np.nan_to_num(pendiente_V, nan=0)
        
        # Estadísticas avanzadas R(t)
        r_mean_post_max = np.nan_to_num(np.mean(valores_resistencia[I_R_max:]), nan=0)
        resistencia_inicial = np.nan_to_num(r0, nan=2000) # nan=2000? Mantenido del original
        r_mean = np.nan_to_num(np.mean(valores_resistencia[:-1]), nan=0)
        
        # Rangos
        rango_r_beta_alfa = np.nan_to_num(resistencia_max - r0, nan=0)
        rango_r_e_beta = np.nan_to_num(r_e - resistencia_max, nan=0)
        rango_t_e_beta = np.nan_to_num(t_e - t_R_max, nan=0)
        
        desv_R = np.nan_to_num(np.std(valores_resistencia[:I_R_max]), nan=0)
        
        # Pendientes
        pendientes = calcular_pendiente(valores_resistencia, t_soldadura)
        pendientes_post_max = pendientes[I_R_max:] # Corrección: Iniciar desde el pico
        pendientes_negativas_post = sum(1 for p in pendientes_post_max if p < 0)
        
        # Áreas (pre/post pico)
        valores_resistencia_hasta_R_max = valores_resistencia[:I_R_max + 1]
        valores_tiempo_hasta_R_max = t_soldadura[:I_R_max + 1]
        area_pre_mitad = np.nan_to_num(np.trapz(valores_resistencia_hasta_R_max, valores_tiempo_hasta_R_max), nan=0)
        
        valores_resistencia_desde_R_max = valores_resistencia[I_R_max:]
        valores_tiempo_desde_R_max = t_soldadura[I_R_max:]
        area_post_mitad = np.nan_to_num(np.trapz(valores_resistencia_desde_R_max, valores_tiempo_desde_R_max), nan=0)
        
        try:
            desv_pre_mitad_t = np.nan_to_num(np.std(valores_resistencia_hasta_R_max), nan=0)
        except ValueError:
            desv_pre_mitad_t = 0
            
        # Derivadas
        primera_derivada, segunda_derivada, tercera_derivada = calcular_derivadas(valores_resistencia, t_soldadura)
        
        try:
            max_curvatura = np.nan_to_num(np.max(np.abs(segunda_derivada)), nan=0)
            puntos_inflexion = np.where(np.diff(np.sign(segunda_derivada)))[0]
            num_puntos_inflexion = np.nan_to_num(len(puntos_inflexion), nan=0)
            max_jerk = np.nan_to_num(np.max(np.abs(tercera_derivada)), nan=0)
        except ValueError:
            max_curvatura, num_puntos_inflexion, max_jerk = 0, 0, 0
            
        # Estadísticas de distribución
        try:
            mediana = np.nan_to_num(np.median(valores_resistencia), nan=0)
            varianza = np.nan_to_num(np.var(valores_resistencia), nan=0)
            rango_intercuartilico = np.nan_to_num((np.percentile(valores_resistencia, 75) - np.percentile(valores_resistencia, 25)), nan=0)
            asimetria = np.nan_to_num(skew(valores_resistencia), nan=0)
            curtosis = np.nan_to_num(kurtosis(valores_resistencia), nan=0)
        except (ValueError, IndexError):
            mediana, varianza, rango_intercuartilico, asimetria, curtosis = 0, 0, 0, 0, 0
            
        # Picos y valles
        valores_resistencia_np = np.array(valores_resistencia)
        picos, _ = find_peaks(valores_resistencia_np, height=0)
        valles, _ = find_peaks(-valores_resistencia_np)
        num_picos = np.nan_to_num(len(picos), nan=0)
        num_valles = np.nan_to_num(len(valles), nan=0)
        
        # Pendiente OLS (Mínimos cuadrados)
        # Advertencia: La fórmula original para 'denominador' parece ser para la 
        # regresión de Y sobre X, no X sobre Y. Se mantiene la fórmula original.
        t_mean = np.nan_to_num(np.mean(t_soldadura), nan=0)
        r_mean_ols = np.nan_to_num(np.mean(valores_resistencia), nan=0)
        numerador = sum((r_mean_ols - ri) * (t_mean - ti) for ri, ti in zip(valores_resistencia, t_soldadura))
        denominador = sum((r_mean_ols - ri) ** 2 for ri in valores_resistencia)
        
        m_min_cuadrados = 0 if denominador == 0 else (numerador / denominador)

        # --- 6. Ensamblar vector de características y etiqueta ---
        X_calculado.append([
            float(rango_r_beta_alfa),         # Feature 0
            float(rango_t_e_beta),            # Feature 1
            float(rango_r_e_beta),            # Feature 2
            float(resistencia_inicial),       # Feature 3
            float(k4),                        # Feature 4
            float(k3),                        # Feature 5
            float(rango_intercuartilico),     # Feature 6
            float(desv_pre_mitad_t),          # Feature 7
            float(resistencia_ultima),        # Feature 8
            float(desv),                      # Feature 9
            float(pendiente_V),               # Feature 10
            float(rms),                       # Feature 11
            float(rango_rmax_rmin),           # Feature 12
            float(r_mean_post_max),           # Feature 13
            float(r_mean),                    # Feature 14
            float(desv_R),                    # Feature 15
            float(pendientes_negativas_post), # Feature 16
            float(rango_tiempo_max_min),      # Feature 17
            float(area_bajo_curva),           # Feature 18
            float(area_pre_mitad),            # Feature 19
            float(area_post_mitad),           # Feature 20
            float(max_curvatura),             # Feature 21
            float(num_puntos_inflexion),      # Feature 22
            float(max_jerk),                  # Feature 23
            float(mediana),                   # Feature 24
            float(varianza),                  # Feature 25
            float(asimetria),                 # Feature 26
            float(curtosis),                  # Feature 27
            float(num_picos),                 # Feature 28
            float(num_valles),                # Feature 29
            float(q),                         # Feature 30
            float(m_min_cuadrados)            # Feature 31
        ])
        
        # Añadir la etiqueta
        valor_i = int(new_df.loc[i, "Etiqueta datos"])
        y_calculado.append(valor_i)
        
    print("Cálculo de features completado.")
    
    return np.array(X_calculado), np.array(y_calculado)


# ==============================================================================
# 4. FUNCIONES DEL PIPELINE DE MACHINE LEARNING
# ==============================================================================

def paso_1_cargar_y_preparar_datos(ruta_csv_defecto, feature_names):
    """
    Orquesta la carga de datos y la creación de los DataFrames X e y.
    
    Args:
        ruta_csv_defecto (str): Ruta al archivo CSV.
        feature_names (list): Lista de 32 nombres para las columnas de X.

    Returns:
        tuple: (X, y) como DataFrames de Pandas, o (None, None) si falla.
    """
    X_raw, y_raw = extraer_features_fila_por_fila(
        preprocesar_dataframe_inicial(
            leer_archivo(ruta_csv_defecto)
        )
    )
    
    if X_raw.size == 0:
        print("No se cargaron datos. Terminando.")
        return None, None
        
    # Convertir a DataFrames de Pandas para mejor manejo
    X = pd.DataFrame(X_raw, columns=feature_names)
    X = X.applymap(lambda x: round(x, 4))
    y = pd.Series(y_raw, name="Etiqueta_Defecto")

    print("\n--- Resumen de Datos Cargados ---")
    print(f"Total de muestras: {len(X)}")
    print(f"Número de características: {X.shape[1]}")
    print(f"Distribución de clases:\n{y.value_counts(normalize=True)}")
    print("----------------------------------\n")
    
    return X, y

def paso_2_escalar_y_dividir_datos(X, y, test_size, random_state):
    """
    Divide los datos en train/test y los escala usando StandardScaler.
    El escalador se ajusta SÓLO en los datos de entrenamiento.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Guardar nombres de columnas para reconstruir DataFrames
    train_cols = X_train.columns
    
    print("Escalando datos...")
    scaler = StandardScaler()
    
    # Ajustar SÓLO en X_train
    X_train_scaled = scaler.fit_transform(X_train)
    # Transformar X_test con el scaler ya ajustado
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de nuevo a DataFrames
    X_train = pd.DataFrame(X_train_scaled, columns=train_cols)
    X_test = pd.DataFrame(X_test_scaled, columns=train_cols)
    
    print("Datos divididos y escalados.")
    return X_train, X_test, y_train, y_test, scaler

def paso_3_entrenar_modelo(X_train, y_train, n_splits, n_iter, fbeta, random_state):
    """
    Configura y ejecuta RandomizedSearchCV para encontrar el mejor
    RandomForestClassifier, optimizando para F2-Score.
    
    Returns:
        RandomizedSearchCV: El objeto de búsqueda ajustado.
    """
    print("Iniciando búsqueda de hiperparámetros para Random Forest...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f2_scorer = make_scorer(fbeta_score, beta=fbeta)

    # 1. Definir el modelo base
    modelo_rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")

    # 2. Definir el espacio de búsqueda de parámetros
    param_dist = {
        'n_estimators': randint(100, 400),
        'max_depth': [3, 4, 5, 7, 10, None],
        'min_samples_leaf': [3, 5, 10, 15]
    }

    # 3. Configurar la Búsqueda Aleatoria
    search_cv = RandomizedSearchCV(
        estimator=modelo_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=skf,
        scoring=f2_scorer,
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )

    # 4. Entrenar
    search_cv.fit(X_train, y_train)
    
    print("Entrenamiento de Random Forest completado.")
    print(f"Mejores parámetros encontrados: {search_cv.best_params_}")
    print(f"Mejor score F2 (en CV): {search_cv.best_score_:.4f}")
    
    return search_cv

def paso_4_evaluar_importancia_y_umbral_defecto(mejor_modelo, X_test, y_test, feature_names):
    """
    Grafica la importancia de las características y la matriz de confusión
    con el umbral por defecto (0.5).
    """
    # --- 1. Importancia de Características ---
    importancias = mejor_modelo.feature_importances_
    df_importancias = pd.DataFrame({
        'predictor': feature_names,
        'importancia': importancias
    }).sort_values(by='importancia', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_importancias.predictor, df_importancias.importancia)
    plt.yticks(size=8)
    ax.set_xlabel('Importancia de la Característica')
    ax.set_ylabel('Variable Predictora')
    ax.set_title('Importancia de Características del Random Forest')
    plt.tight_layout()
    plt.show()

    # --- 2. Matriz de Confusión (Umbral 0.5) ---
    predicciones_defecto = mejor_modelo.predict(X_test)
    matriz_confusion = confusion_matrix(y_test, predicciones_defecto)
    
    # *** CORRECCIÓN ***: El título original era incorrecto.
    titulo = "Matriz de Confusión - RANDOM FOREST (Umbral = 0.5)"
    
    _plot_confusion_matrix(matriz_confusion, titulo)

def paso_5_optimizar_umbral(mejor_modelo, X_train, y_train, n_splits, precision_minima, random_state):
    """
    Busca el umbral de decisión óptimo que maximiza el Recall
    cumpliendo con una Precisión mínima, usando validación cruzada.
    """
    print(f"\nOptimizando umbral para: MAX(Recall) sujeto a Precision >= {precision_minima}...")

    cv_thresh = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    lista_umbrales = np.linspace(0.01, 0.7, 1000) 

    optimal_threshold = 0.5 # Valor de fallback
    best_recall = -1
    best_precision_at_best_recall = 0

    # Usamos validación cruzada para encontrar el umbral
    for train_idx, val_idx in cv_thresh.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Re-entrenar el mejor modelo en el fold de entrenamiento
        modelo_cv_pipe = mejor_modelo.fit(X_train_cv, y_train_cv)
        
        # Obtener probabilidades en el fold de validación
        y_pred_proba_cv = modelo_cv_pipe.predict_proba(X_val_cv)[:, 1]

        for thresh in lista_umbrales:
            y_pred_thresh = np.where(y_pred_proba_cv >= thresh, 1, 0)
            
            prec = precision_score(y_val_cv, y_pred_thresh, zero_division=0)
            rec = recall_score(y_val_cv, y_pred_thresh, zero_division=0)

            # 1. ¿Cumple la condición de precisión?
            if prec >= precision_minima:
                # 2. Si la cumple, ¿es el mejor RECALL que hemos visto?
                if rec > best_recall:
                    best_recall = rec
                    optimal_threshold = thresh
                    best_precision_at_best_recall = prec

    # --- Comprobación y Fallback ---
    # *** CORRECCIÓN ***: Eliminado el "número mágico" 0.5115.
    if best_recall == -1:
        print(f"¡ADVERTENCIA! No se encontró NINGÚN umbral que cumpla 'Precision >= {precision_minima}'.")
        print("El modelo no puede satisfacer este requisito con los datos de CV.")
        print("Como fallback, se usará el umbral por defecto de 0.5.")
        optimal_threshold = 0.5
    else:
        print("¡Éxito! Se encontró un umbral que cumple los requisitos.")
        print(f"   -> Umbral óptimo: {optimal_threshold:.4f}")
        print(f"   -> Recall resultante (en CV): {best_recall:.4f}")
        print(f"   -> Precision resultante (en CV): {best_precision_at_best_recall:.4f}")
        
    return optimal_threshold

def paso_6_evaluacion_final_y_guardado(mejor_modelo, X_test, y_test, scaler, optimal_threshold):
    """
    Realiza la evaluación final en el conjunto de Test usando el umbral
    optimizado y guarda los artefactos del modelo.
    """
    print("\n--- Evaluación Final en Conjunto de Prueba (Test Set) ---")
    
    # Obtener probabilidades del Test Set
    predicciones_test_proba = mejor_modelo.predict_proba(X_test)[:, 1]
    
    # Aplicar el umbral óptimo
    predicciones_test_binarias = np.where(predicciones_test_proba >= optimal_threshold, 1, 0)

    # --- 1. Reporte de Clasificación ---
    print("\nReporte de Clasificación (Test Set):")
    target_names = ['0: Sin Defecto', '1: Con Defecto (Pegado)']
    print(classification_report(y_test, predicciones_test_binarias, target_names=target_names))

    # --- 2. Matriz de Confusión (Umbral Óptimo) ---
    matriz_confusion_opt = confusion_matrix(y_test, predicciones_test_binarias)
    titulo = f"Matriz de Confusión - RANDOM FOREST (Umbral Óptimo = {optimal_threshold:.4f})"
    _plot_confusion_matrix(matriz_confusion_opt, titulo)

    # --- 3. Curva ROC ---
    # *** CORRECCIÓN ***: Se usa `predicciones_test_proba` (probabilidades)
    # en lugar de `predicciones_test_binarias` (0s y 1s).
    fpr, tpr, _ = metrics.roc_curve(y_test, predicciones_test_proba)
    auc_score = metrics.roc_auc_score(y_test, predicciones_test_proba)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Clasificador Aleatorio (AUC = 0.5)")
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC (Test Set)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # --- 4. Guardar Artefactos del Modelo ---
    print("\nGuardando modelo, scaler y umbral en 'modelo_con_umbral_PEGADOS.pkl'...")
    # Es VITAL guardar el 'scaler' para predecir en datos nuevos.
    artefactos_modelo = {
        "modelo": mejor_modelo,
        "scaler": scaler,
        "umbral": optimal_threshold,
        "feature_names": FEATURE_NAMES
    }
    with open('modelo_con_umbral_PEGADOS.pkl', 'wb') as f:
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


# ==============================================================================
# 5. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

def main():
    """
    Función principal que orquesta todo el pipeline de ML.
    """
    # PASO 1: Cargar y procesar los datos crudos
    X, y = paso_1_cargar_y_preparar_datos(RUTA_CSV_POR_DEFECTO, FEATURE_NAMES)
    
    if X is None:
        return  # Salir si la carga de datos falló

    # PASO 2: Dividir y escalar los datos
    X_train, X_test, y_train, y_test, scaler = paso_2_escalar_y_dividir_datos(
        X, y, TEST_SIZE_RATIO, RANDOM_STATE_SEED
    )

    # PASO 3: Entrenar el modelo con búsqueda de hiperparámetros
    search_cv = paso_3_entrenar_modelo(
        X_train, y_train, 
        N_SPLITS_CV, N_ITER_RANDOM_SEARCH, FBETA_BETA, RANDOM_STATE_SEED
    )
    
    mejor_modelo = search_cv.best_estimator_

    # PASO 4: Evaluación inicial (Importancia de features, matriz con umbral 0.5)
    paso_4_evaluar_importancia_y_umbral_defecto(
        mejor_modelo, X_test, y_test, FEATURE_NAMES
    )

    # PASO 5: Optimizar el umbral de decisión
    optimal_threshold = paso_5_optimizar_umbral(
        mejor_modelo, X_train, y_train, 
        N_SPLITS_CV, PRECISION_MINIMA, RANDOM_STATE_SEED
    )

    # PASO 6: Evaluación final (en Test set) y guardado del modelo
    paso_6_evaluacion_final_y_guardado(
        mejor_modelo, X_test, y_test, scaler, optimal_threshold
    )


if __name__ == "__main__":
    # Esta línea asegura que la función main() solo se ejecute
    # cuando el script se corre directamente.
    main()