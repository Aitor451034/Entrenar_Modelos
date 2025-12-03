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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc, fbeta_score, make_scorer, classification_report, confusion_matrix,
    precision_score, recall_score, roc_curve, roc_auc_score
)
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier # Para usarlo como filtro en el selector
from sklearn.feature_selection import RFE


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
PRECISION_MINIMA = 0.7


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
    *** LÓGICA CENTRAL: SMOTE + RandomForest ***
    Configura y ejecuta GridSearchCV en un pipeline de SMOTE y RandomForest.
    """
    print("Iniciando búsqueda de hiperparámetros para SMOTE + RandomForest...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f2_scorer = make_scorer(fbeta_score, beta=fbeta)

    # 1. Definir el modelo RandomForest
    # Se omite 'class_weight' para dejar que SMOTE haga el trabajo de balanceo.
    modelo_hibrido_rf = RandomForestClassifier(random_state=random_state,class_weight='balanced')

    # 2. Definir el Pipeline de Imbalanced-learn
    # 
    pipeline_rf = ImbPipeline([
        ('scaler', StandardScaler()),           # 1. Escalar
        ('selector', RFE(           # 3. Seleccionar Features
            RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
            step=1,  # Elimina de uno en uno 
            verbose=0
        )),
        ('model', modelo_hibrido_rf)               # Paso 4: Modelo
    ])

    # 3. Definir el GRID de parámetros para el pipeline
    # (Los nombres deben incluir el prefijo 'model__')
    param_grid_rf = {
        'model__n_estimators': [200, 300, 400],
        'model__max_depth': [5, 7, 10],         # Se quita 'None' para evitar overfitting
        'model__min_samples_leaf': [3, 5, 10],    # > 1 fuerza a generalizar
        'model__min_samples_split': [2, 5, 10],
        'model__max_features': ['sqrt', 'log2'],
        'selector__estimator__max_features': [15,20,25]              # --- Parámetros del Selector ---#
    }
    
    total_combinaciones = np.prod([len(v) for v in param_grid_rf.values()])
    print(f"GridSearchCV (SMOTE+RF) probará {total_combinaciones} combinaciones.")
    print("Entrenando... (Esto puede tardar)")

    # 4. Configurar y ejecutar la Búsqueda (GridSearchCV)
    search_cv = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid_rf,
        cv=skf,
        scoring=f2_scorer,
        n_jobs=-1,
        verbose=2
    )

    search_cv.fit(X_train, y_train)
    
    mejor_modelo = search_cv.best_estimator_
    print("Entrenamiento (GridSearchCV) de SMOTE + RandomForest completado.")
    print(f"Mejores parámetros encontrados: {search_cv.best_params_}")
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
    print("\nImportancia de las 32 características (features) para el modelo SMOTE + RandomForest:")
    print(df_importancias.sort_values(by='importancia', ascending=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_importancias.predictor, df_importancias.importancia)
    plt.yticks(size=8)
    ax.set_xlabel('Importancia de la Característica')
    ax.set_ylabel('Variable Predictora')
    # *** CORRECCIÓN ***: Título del gráfico
    ax.set_title('Importancia de Características (SMOTE + RandomForest)')
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
    titulo = f"Matriz de Confusión - SMOTE + RF (Umbral Óptimo = {optimal_threshold:.4f})"
    _plot_confusion_matrix(matriz_confusion_opt, titulo)

    # --- 3. Curva ROC ---
    # 
    # *** CORRECCIÓN CRÍTICA ***: Usar probabilidades, no predicciones binarias.
    fpr, tpr, _ = metrics.roc_curve(y_test, predicciones_test_proba)
    auc_score = metrics.roc_auc_score(y_test, predicciones_test_proba)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"SMOTE + RF (AUC = {auc_score:.4f})")
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
    print("\nGuardando pipeline COMPLETO (Scaler+SMOTE+Selector+Modelo) y umbral...")
    
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