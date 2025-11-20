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
from sklearn import metrics
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier # Para usarlo como filtro en el selector


# --- NUEVAS BIBLIOTECAS: Imbalanced-learn ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


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
PRECISION_MINIMA = 0.68


# ==============================================================================
# 3. FUNCIONES DE CARGA Y EXTRACCIÓN DE CARACTERÍSTICAS
# ==============================================================================
# (Funciones idénticas a las versiones anteriores, colapsadas por brevedad)

def leer_archivo(ruta_csv_defecto):
    """Lee un archivo CSV con datos de soldadura."""
    print("Abriendo archivo ...")
    try:
        df = pd.read_csv(ruta_csv_defecto, encoding="utf-8", sep=";", on_bad_lines="skip", header=None, quotechar='"', decimal=",", skiprows=3)
        print("¡Archivo CSV leído correctamente desde la ruta por defecto!")
        return df
    except FileNotFoundError:
        print("No se ha encontrado el archivo en la ruta por defecto. Abriendo diálogo...")
        root = tk.Tk()
        root.withdraw()
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
# 4. FUNCIONES DEL PIPELINE DE MACHINE LEARNING
# ==============================================================================

def paso_1_cargar_y_preparar_datos(ruta_csv_defecto, feature_names):
    """Orquesta la carga de datos y la creación de los DataFrames X e y."""
    df_raw = leer_archivo(ruta_csv_defecto)
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
    modelo_hibrido_rf = RandomForestClassifier(random_state=random_state)

    # 2. Definir el Pipeline de Imbalanced-learn
    # 
    pipeline_rf = ImbPipeline([
        ('scaler', StandardScaler()),           # 1. Escalar
        ('smote', BorderlineSMOTE(random_state=random_state)), # Paso 2: Sobremuestreo
        ('selector', RFE(           # 3. Seleccionar Features
            RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1,),
            step=1,  # Elimina 1 a 1 (máxima precisión)
            verbose=0
        )),
        ('model', modelo_hibrido_rf)               # Paso 4: Modelo
    ])

    # 3. Definir el GRID de parámetros para el pipeline
    # (Los nombres deben incluir el prefijo 'model__')
    param_grid_rf = {
        'smote': [BorderlineSMOTE(random_state=random_state)],
        'model__n_estimators': [200, 300, 400],
        'model__max_depth': [5, 7, 10],         # Se quita 'None' para evitar overfitting
        'model__min_samples_leaf': [3, 5, 10],    # > 1 fuerza a generalizar
        'model__min_samples_split': [2, 5, 10],
        'model__max_features': ['sqrt', 'log2'],
        'selector__estimator__max_features': [15 ,20 ,25]              # --- Parámetros del Selector ---#
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


if __name__ == "__main__":
    main()