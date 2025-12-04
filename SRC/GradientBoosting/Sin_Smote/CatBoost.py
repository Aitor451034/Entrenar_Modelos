    
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
    b. Selecciona las mejores características con SelectFromModel (usando CatBoost).
    c. Entrena un modelo CatBoostClassifier con pesos de clase.
4.  Búsqueda aleatoria de hiperparámetros (RandomizedSearchCV) en el pipeline.
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
from scipy.signal import find_peaks,savgol_filter #Filtro de Savgol

# --- Componentes de Scikit-learn ---
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_predict, RandomizedSearchCV
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
RANDOM_STATE_SEED = 42
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
    """
    Versión OPTIMIZADA Y CORREGIDA.
    Calcula las 32 características físicas y estadísticas filtrando el ruido.
    """
    X_calculado = []
    y_calculado = []
    
    print(f"Procesando {len(new_df)} puntos de soldadura (Algoritmo Corregido)...")

    for i in new_df.index:
        try:
            # --- 1. LECTURA Y LIMPIEZA DE SERIES ---
            str_volt = new_df.loc[i, "Voltajes inst."]
            str_corr = new_df.loc[i, "Corrientes inst."]
            
            if pd.isna(str_volt) or pd.isna(str_corr):
                print(f"Fila {i}: Datos nulos. Saltando.")
                continue
                
            # Convertir a float (Sin redondear agresivamente para no perder info)
            raw_volt = np.array([float(v) for v in str_volt.split(';') if v.strip()])
            raw_corr = np.array([float(v) for v in str_corr.split(';') if v.strip()])
            
            ns = int(new_df.loc[i, "Ns"])
            ts2 = int(new_df.loc[i, "Ts2"])
            t_soldadura = np.linspace(0, ts2, ns + 1)
            
            # Recorte de seguridad (igualar longitudes)
            min_len = min(len(t_soldadura), len(raw_volt), len(raw_corr))
            if min_len < 10: continue 
            
            t_soldadura = t_soldadura[:min_len]
            raw_volt = raw_volt[:min_len]
            raw_corr = raw_corr[:min_len]

            # --- 2. CÁLCULO DE RESISTENCIA (FILTRADO) ---
            # CORRECCIÓN: Evitar infinitos si corriente es ~0
            valores_resistencia = np.divide(raw_volt, raw_corr, out=np.zeros_like(raw_volt), where=np.abs(raw_corr)>0.5)
            
            # CORRECCIÓN: Filtro Savitzky-Golay (Vital para derivadas limpias)
            window = min(11, len(valores_resistencia) if len(valores_resistencia)%2!=0 else len(valores_resistencia)-1)
            if window > 3:
                r_smooth = savgol_filter(valores_resistencia, window_length=window, polyorder=3)
            else:
                r_smooth = valores_resistencia

            # --- 3. EXTRACCIÓN DE PUNTOS CLAVE ---
            idx_max = np.argmax(r_smooth)
            idx_min = np.argmin(r_smooth)
            
            resistencia_max = r_smooth[idx_max]
            t_R_max = t_soldadura[idx_max]
            
            r0 = r_smooth[0]
            r_e = r_smooth[-1]
            t_e = t_soldadura[-1]
            resistencia_min = np.min(r_smooth)
            t_min = t_soldadura[idx_min] # Necesario para rango tiempo max-min

            # --- 4. CÁLCULO DE ENERGÍA (CORREGIDO) ---
            # CORRECCIÓN: kA * 1000 = Amperios (Antes tenías * 10)
            i_amperios = raw_corr * 1000.0 
            v_reales = raw_volt / 100.0    # Ajusta si tu sensor ya da voltios reales
            t_segundos = t_soldadura / 1000.0
            
            potencia = v_reales * i_amperios
            q_joules = np.trapz(potencia, x=t_segundos) # Energía real en Joules

            # --- 5. DERIVADAS Y EVENTOS FÍSICOS ---
            # Calculamos derivadas sobre la señal SUAVIZADA
            d1 = np.gradient(r_smooth, t_soldadura)
            d2 = np.gradient(d1, t_soldadura)
            d3 = np.gradient(d2, t_soldadura)
            
            max_curvatura = np.max(np.abs(d2))
            max_jerk = np.max(np.abs(d3))
            puntos_inflexion = np.sum(np.diff(np.sign(d2)) != 0)
            
            # Picos y Valles
            picos, _ = find_peaks(r_smooth)
            valles, _ = find_peaks(-r_smooth)
            num_picos = len(picos)
            num_valles = len(valles)

            # --- 6. CÁLCULOS ESTADÍSTICOS Y PENDIENTES ---
            
            # Pendiente Mínimos Cuadrados (CORREGIDA)
            t_mean = np.mean(t_soldadura)
            r_mean = np.mean(r_smooth)
            # El denominador debe ser la varianza de T, no de R
            numerador = np.sum((r_smooth - r_mean) * (t_soldadura - t_mean))
            denominador = np.sum((t_soldadura - t_mean)**2) 
            m_ols = numerador / denominador if denominador != 0 else 0

            # Pendiente Voltaje
            idx_v_max = np.argmax(raw_volt)
            pendiente_V = 0
            if idx_v_max < len(raw_volt) - 1:
                dt_v = t_soldadura[idx_v_max] - t_e
                if dt_v != 0:
                     pendiente_V = (raw_volt[idx_v_max] - raw_volt[-1]) / dt_v

            # Pendientes K3 y K4
            k3 = ((resistencia_max - r0) / t_R_max * 100) if t_R_max > 0 else 0
            delta_t_post = t_e - t_R_max
            k4 = ((r_e - resistencia_max) / delta_t_post * 100) if delta_t_post > 0 else 0

            # Pendientes negativas post-pico
            pendientes_post = d1[idx_max:]
            num_negativas = np.sum(pendientes_post < 0)

            # Estadísticas
            desv = np.std(r_smooth)
            rms = np.sqrt(np.mean(r_smooth**2))
            mediana = np.median(r_smooth)
            varianza = np.var(r_smooth)
            iqr = np.percentile(r_smooth, 75) - np.percentile(r_smooth, 25)
            asim = skew(r_smooth) if len(r_smooth) > 2 else 0
            curt = kurtosis(r_smooth) if len(r_smooth) > 2 else 0
            
            # Desviaciones parciales
            desv_pre_mitad_t = np.std(r_smooth[:len(r_smooth)//2])
            desv_R = np.std(r_smooth[:idx_max+1]) # Pre-pico
            r_mean_post_max = np.mean(r_smooth[idx_max:]) if idx_max < len(r_smooth) else 0

            # Áreas
            area_total = np.trapz(r_smooth, t_soldadura)
            idx_mitad = len(t_soldadura) // 2
            area_pre_mitad = np.trapz(r_smooth[:idx_mitad], t_soldadura[:idx_mitad])
            area_post_mitad = area_total - area_pre_mitad
            
            # Rangos
            rango_r_beta_alfa = resistencia_max - r0
            rango_r_e_beta = r_e - resistencia_max
            rango_t_e_beta = t_e - t_R_max
            rango_rmax_rmin = resistencia_max - resistencia_min
            rango_tiempo_max_min = t_R_max - t_min

            # --- 7. ENSAMBLAJE FINAL (ORDEN DE TU CÓDIGO ORIGINAL) ---
            fila_features = [
                float(rango_r_beta_alfa),       # 1
                float(rango_t_e_beta),          # 2
                float(rango_r_e_beta),          # 3
                float(r0),                      # 4 (Resistencia Inicial)
                float(k4),                      # 5
                float(k3),                      # 6
                float(iqr),                     # 7 (Rango intercuartílico)
                float(desv_pre_mitad_t),        # 8
                float(r_e),                     # 9 (Resistencia Final)
                float(desv),                    # 10
                float(pendiente_V),             # 11
                float(rms),                     # 12
                float(rango_rmax_rmin),         # 13
                float(r_mean_post_max),         # 14
                float(r_mean),                  # 15
                float(desv_R),                  # 16 (Desv Pre-Pico)
                float(num_negativas),           # 17
                float(rango_tiempo_max_min),    # 18
                float(area_total),              # 19
                float(area_pre_mitad),          # 20
                float(area_post_mitad),         # 21
                float(max_curvatura),           # 22
                float(puntos_inflexion),        # 23
                float(max_jerk),                # 24
                float(mediana),                 # 25
                float(varianza),                # 26
                float(asim),                    # 27
                float(curt),                    # 28
                float(num_picos),               # 29
                float(num_valles),              # 30
                float(q_joules),                # 31 (Energía corregida)
                float(m_ols)                    # 32 (Mínimos Cuadrados corregido)
            ]
            
            # Limpieza final de NaNs o Infinitos
            fila_features = np.nan_to_num(fila_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_calculado.append(fila_features)
            y_calculado.append(int(new_df.loc[i, "Etiqueta datos"]))

        except Exception as e:
            print(f"Error en fila {i}: {e}")
            continue

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
    *** LÓGICA CENTRAL: Pipeline Completo (Scaler + Selector + CatBoost) ***
    Configura y ejecuta GridSearchCV en un pipeline completo para prevenir Data Leakage.
    
    Este pipeline encadena tres pasos:
    1. Escalado de características (StandardScaler).
    2. Selección de características (SelectFromModel con CatBoost).
    3. Clasificación (CatBoostClassifier).
    
    La búsqueda aleatoria (RandomizedSearchCV) optimiza hiperparámetros para maximizar F2-score.
    """
    print("Iniciando búsqueda aleatoria de hiperparámetros para Pipeline Completo (Scaler + Selector + CatBoost)...")
    
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
        
        # Paso 2: Seleccionar las mejores características usando SelectFromModel.
        # Se entrena un CatBoost para obtener la importancia de cada feature y se seleccionan las mejores.
        # Es más rápido y consistente que usar RFE con otro modelo como RandomForest.
        ('selector', SelectFromModel(
            CatBoostClassifier(
                random_seed=random_state,
                verbose=False,
                class_weights=class_weights,
                #task_type="GPU", # Usar GPU también para la selección acelera el proceso
                #devices='0'
            )
        )),
        
        # Paso 3: Modelo clasificador CatBoost (Gradient Boosting)
        # CatBoost es robusto con features categóricas y maneja el desbalance de clases bien
        ('model', CatBoostClassifier(
            # --- Parámetros Fijos ---
            loss_function="Logloss",        # Función de pérdida para clasificación binaria.
            eval_metric="Recall",           # Métrica a monitorear durante el entrenamiento.
            bootstrap_type='Bernoulli',     # Tipo de bootstrapping.
            random_seed=random_state,       # Semilla para reproducibilidad.
            od_type="Iter",                 # Habilita la detección de sobreajuste.
            verbose=False,                  # No mostrar logs detallados del entrenamiento.
            class_weights=class_weights,    # Pesos para manejar el desbalance de clases.
            
            # --- PARÁMETROS PARA ACTIVAR LA GPU ---
            #task_type="GPU",                # ¡CLAVE! Indica a CatBoost que use la GPU.
            #devices='0'                     # Especifica el índice de la GPU a usar (normalmente '0' para la primera).
        ))
    ])

    # --- DEFINICIÓN DE LA GRILLA DE HIPERPARÁMETROS ---
    # RandomizedSearchCV probará una muestra aleatoria de las combinaciones de estos valores.
    # Los nombres con prefijo 'model__' son parámetros del modelo CatBoost
    param_grid_cb = {
        # --- Parámetros del modelo CatBoost para reducir overfitting ---
        'model__depth': [4, 5, 6],                          # Profundidad de los árboles (menor = menos complejo)
        'model__l2_leaf_reg': [3, 5, 7, 10],                 # Regularización L2 (mayor = menos overfitting)
        'model__learning_rate': [0.03, 0.05, 0.08],          # Tasa de aprendizaje
        'model__subsample': [0.7, 0.8, 0.9],                 # Porcentaje de muestras para entrenar cada árbol
        'model__min_data_in_leaf': [5, 10, 20],              # Muestras mínimas en hojas (previene overfitting en ruido)
        'model__iterations': [100, 200, 300],                # Más iteraciones, con early stopping

        # El parámetro 'selector__max_features' se ha eliminado.
        # Ahora, SelectFromModel usará su umbral por defecto ('mean'),
        # seleccionando automáticamente las features con importancia superior a la media.
        # Esto responde a la petición de no fijar un número bajo de características
        # y dejar que el modelo elija las más relevantes, descartando las ruidosas.
    }
    
    # Número de combinaciones aleatorias a probar. Es mucho más rápido que GridSearchCV.
    n_iter_search = 100
    print(f"RandomizedSearchCV (CatBoost) probará {n_iter_search} combinaciones aleatorias de hiperparámetros.")
    print("Entrenando... (Esto puede tardar varios minutos)\n")

    # --- EJECUTAR BÚSQUEDA ALEATORIA (RandomizedSearchCV) ---
    # Es mucho más eficiente que GridSearchCV para espacios de búsqueda grandes.
    # Selecciona la combinación que maximiza la métrica F2-score en validación cruzada.
    search_cv = RandomizedSearchCV(
        estimator=pipeline_cb,
        param_distributions=param_grid_cb, # Ojo: el parámetro se llama param_distributions
        n_iter=n_iter_search,         # Número de combinaciones a probar
        cv=skf,                       # Usar validación cruzada estratificada
        scoring=f2_scorer,            # Métrica a optimizar (F2-score)
        # Usar todos los núcleos. Al haber menos iteraciones, el riesgo de
        # cuello de botella es menor y la paralelización es beneficiosa.
        n_jobs=-1,
        verbose=2,                    # Mostrar progreso
        refit=True,                   # Reentrenar con mejores parámetros en todo el set
        random_state=random_state     # Para que la búsqueda aleatoria sea reproducible
    )

    # Entrenar el pipeline con la búsqueda de hiperparámetros
    search_cv.fit(X_train, y_train)
    
    # Extraer el modelo entrenado con los mejores parámetros
    mejor_modelo = search_cv.best_estimator_
    
    print("\n" + "="*70)
    print("Entrenamiento (RandomizedSearchCV) de CatBoost completado.")
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
    #    a) StandardScaler: Normaliza características
    #    b) SelectFromModel: Selecciona las N mejores características usando CatBoost
    #    c) CatBoost: Modelo de Gradient Boosting (árbol mejorado iterativamente)
    #
    # 2. Ejecuta RandomizedSearchCV que prueba 50 combinaciones aleatorias de:
    #    - Profundidad del árbol: [4, 6, 8] 
    #    - Regularización L2: [3, 7]
    #    - Tasa de aprendizaje: [0.03, 0.06]
    #    - Subsample (% muestras por árbol): [0.7, 0.85]
    #    - Mínimo de datos en hojas: [1, 5]
    #    - Número de features a seleccionar: [3, 5, 8]
    #    - Iteraciones: [50, 100, 150]
    #
    # 3. Usa validación cruzada estratificada (5 folds) para evaluar cada combo
    #
    # 4. Selecciona la combinación que MAXIMIZA el F2-score
    #    (F2 da más peso al Recall que a Precision, importante para detectar defectos)
    #
    print("\n" + "="*70)
    print("PASO 3: ENTRENANDO MODELO CON BÚSQUEDA DE HIPERPARÁMETROS (RandomizedSearchCV)")
    print("="*70)
    print("⏳ ADVERTENCIA: Este paso puede tardar varios minutos...")
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