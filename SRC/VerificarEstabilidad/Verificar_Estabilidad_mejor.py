"""
Script Genérico para Verificación de Estabilidad (Bootstrapping).
Soporta: CatBoost, XGBoost, LightGBM y RandomForest.

Este script ejecuta el pipeline completo múltiples veces variando la semilla
aleatoria para medir la desviación estándar de las métricas.
"""

# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import warnings

# --- Modelos ---
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- Utilidades ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- Funciones Científicas (Legacy) ---
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

# Ignorar advertencias
warnings.filterwarnings('ignore')

# ==============================================================================
# 2. CONFIGURACIÓN DE LA PRUEBA
# ==============================================================================

# --------------------------------------------------------------------------
# A) SELECCIONA TU MODELO
# Opciones: 'CatBoost', 'XGBoost', 'LightGBM', 'RandomForest'
# --------------------------------------------------------------------------
TIPO_MODELO = 'CatBoost' 

# --------------------------------------------------------------------------
# B) PEGA TUS MEJORES PARÁMETROS AQUÍ
# (Copia el diccionario que te dio el script de GridSearchCV)
# Nota: Asegúrate de que los parámetros coinciden con el TIPO_MODELO elegido.
# --------------------------------------------------------------------------
MEJORES_PARAMETROS_GRID = {
    # Ejemplo para CatBoost (Reemplázalo con los tuyos):
    'model__depth': 3,
    'model__l2_leaf_reg': 7,
    'model__learning_rate': 0.08,
    'model__subsample': 0.8,
    'selector__n_features_to_select': 25,
    # 'model__colsample_bylevel': 0.8 # (Si usaste GPU, recuerda que este se quitaba)
}

# --------------------------------------------------------------------------
# C) CONFIGURACIÓN DEL BUCLE
# --------------------------------------------------------------------------
N_ITERACIONES_ESTABILIDAD = 30   # Recomendado: 20-30
TEST_SIZE_RATIO = 0.4            # El mismo que en el entrenamiento
METRICA_FOCO = 'Recall'          # 'Recall' o 'Precision'
UMBRAL_DESVIACION_ESTABLE = 0.05 # +/- 5% de variación aceptable


# ==============================================================================
# 3. FUNCIONES DE CARGA (IDÉNTICAS A TU PROYECTO)
# ==============================================================================
# ... (Resumido para no ocupar espacio, la lógica es la misma) ...

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

# ==============================================================================
# 3. FUNCIONES DE CARGA Y EXTRACCIÓN DE CARACTERÍSTICAS
# ==============================================================================
# (Copiadas directamente de tu script anterior para que este sea autónomo)

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

def preprocesar_dataframe_inicial(df):
    """Limpia el DataFrame crudo."""
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
    return new_df

def extraer_features_fila_por_fila(new_df):
    """Itera sobre cada fila y calcula el vector de 32 características."""
    X_calculado, y_calculado = [], []
    print(f"Iniciando cálculo de features para {len(new_df)} puntos...")
    for i in new_df.index:
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
        resistencia_ultima = valores_resistencia[-2]
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
        desv_R = np.nan_to_num(np.std(valores_resistencia[:I_R_max]), nan=0)
        pendientes = calcular_pendiente(valores_resistencia, t_soldadura)
        pendientes_post_max = pendientes[I_R_max:]
        pendientes_negativas_post = sum(1 for p in pendientes_post_max if p < 0)
        valores_resistencia_hasta_R_max, valores_tiempo_hasta_R_max = valores_resistencia[:I_R_max + 1], t_soldadura[:I_R_max + 1]
        area_pre_mitad = np.nan_to_num(np.trapz(valores_resistencia_hasta_R_max, valores_tiempo_hasta_R_max), nan=0)
        valores_resistencia_desde_R_max, valores_tiempo_desde_R_max = valores_resistencia[I_R_max:], t_soldadura[I_R_max:]
        area_post_mitad = np.nan_to_num(np.trapz(valores_resistencia_desde_R_max, valores_tiempo_desde_R_max), nan=0)
        try: desv_pre_mitad_t = np.nan_to_num(np.std(valores_resistencia_hasta_R_max), nan=0)
        except ValueError: desv_pre_mitad_t = 0
        primera_derivada, segunda_derivada, tercera_derivada = calcular_derivadas(valores_resistencia, t_soldadura)
        try:
            max_curvatura = np.nan_to_num(np.max(np.abs(segunda_derivada)), nan=0)
            puntos_inflexion = np.where(np.diff(np.sign(segunda_derivada)))[0]
            num_puntos_inflexion = np.nan_to_num(len(puntos_inflexion), nan=0)
            max_jerk = np.nan_to_num(np.max(np.abs(tercera_derivada)), nan=0)
        except ValueError: max_curvatura, num_puntos_inflexion, max_jerk = 0, 0, 0
        try:
            mediana, varianza = np.nan_to_num(np.median(valores_resistencia), nan=0), np.nan_to_num(np.var(valores_resistencia), nan=0)
            rango_intercuartilico = np.nan_to_num((np.percentile(valores_resistencia, 75) - np.percentile(valores_resistencia, 25)), nan=0)
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
            float(rango_intercuartilico), float(desv_pre_mitad_t), float(resistencia_ultima), float(desv), float(pendiente_V),
            float(rms), float(rango_rmax_rmin), float(r_mean_post_max), float(r_mean), float(desv_R), float(pendientes_negativas_post),
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
    if df_raw is None:
        return None, None
    df_preprocesado = preprocesar_dataframe_inicial(df_raw)
    X_raw, y_raw = extraer_features_fila_por_fila(df_preprocesado)
    
    if X_raw.size == 0:
        print("No se cargaron datos. Terminando.")
        return None, None
        
    X = pd.DataFrame(X_raw, columns=feature_names)
    y = pd.Series(y_raw, name="Etiqueta_Defecto")
    return X, y

# ==============================================================================
# 4. LÓGICA DE SELECCIÓN DE MODELO
# ==============================================================================

def obtener_instancia_modelo(tipo, semilla, params_limpios):
    """Devuelve el objeto del modelo configurado según el nombre."""
    
    if tipo == 'CatBoost':
        return CatBoostClassifier(
            random_seed=semilla,     # CatBoost usa 'random_seed'
            verbose=False,           # Silenciar output
            allow_writing_files=False,
            loss_function="Logloss", # Default estándar
            eval_metric="Recall",
            **params_limpios
        )
    
    elif tipo == 'XGBoost':
        return XGBClassifier(
            random_state=semilla,    # XGB usa 'random_state'
            verbosity=0,             # Silenciar
            use_label_encoder=False,
            eval_metric='logloss',
            **params_limpios
        )
    
    elif tipo == 'LightGBM':
        return LGBMClassifier(
            random_state=semilla,
            verbose=-1,              # Silenciar
            **params_limpios
        )
        
    elif tipo == 'RandomForest':
        return RandomForestClassifier(
            random_state=semilla,
            n_jobs=-1,
            **params_limpios
        )
    
    else:
        raise ValueError(f"Modelo '{tipo}' no reconocido.")

# ==============================================================================
# 5. FUNCIÓN PRINCIPAL DE ESTABILIDAD
# ==============================================================================

def cargar_datos_simulados_o_reales():
    # ESTA FUNCIÓN ES UN PLACEHOLDER. 
    # DEBES USAR TUS FUNCIONES REALES DE CARGA AQUÍ.
    # Si tienes el CSV a mano, el script intentará cargarlo.
    
    # Intento de carga real usando tu lógica (simplificada)
    try:
        df = leer_archivo(RUTA_CSV_POR_DEFECTO)
        if df is None: raise Exception("No CSV")
        # Aquí iría toda tu lógica de preprocesamiento...
        # Como no puedo ejecutar tu preprocesamiento sin el CSV,
        # asegúrate de copiar tus funciones 'paso_1_cargar...' aquí.
        print("AVISO: Debes integrar tus funciones de carga de datos en la sección 3.")
        return None, None 
    except:
        print("Error cargando datos reales. Asegúrate de copiar las funciones de extracción.")
        return None, None

def ejecutar_prueba_estabilidad():
    print(f"--- INICIANDO PRUEBA DE ESTABILIDAD ({TIPO_MODELO}) ---")
    
    # 1. CARGAR DATOS
    X, y = cargar_datos_completos(RUTA_CSV_POR_DEFECTO, FEATURE_NAMES)
    if X is None:
        print("Error cargando datos.")
        return
    
    # 2. LIMPIAR PARÁMETROS
    params_limpios = {k.replace('model__', ''): v for k, v in MEJORES_PARAMETROS_GRID.items()}
    
    # Eliminar parámetros incompatibles con el constructor si es necesario
    # (Por ejemplo, si 'selector__max_features' está en el grid, hay que quitarlo 
    # porque el modelo final no lo entiende, eso era para el RFE)
    claves_a_borrar = [k for k in params_limpios.keys() if 'selector' in k]
    for k in claves_a_borrar:
        del params_limpios[k]

    print(f"Parámetros del modelo: {params_limpios}")

    lista_recalls = []
    lista_precisions = []

    # 3. BUCLE DE ITERACIONES
    for i in range(N_ITERACIONES_ESTABILIDAD):
        seed = i + 42 # Semilla variable
        print(f"Iteración {i+1}/{N_ITERACIONES_ESTABILIDAD}...", end="\r")

        # A. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE_RATIO, random_state=seed, stratify=y
        )

        # B. Instanciar Modelo (Genérico)
        modelo = obtener_instancia_modelo(TIPO_MODELO, seed, params_limpios)

        # C. Pipeline (Scaler + SMOTE + Modelo)
        # Nota: No incluimos RFE aquí para hacer la prueba más rápida y centrarla
        # en la estabilidad del modelo final, pero podrías incluirlo si quieres.
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=seed)),
            ('model', modelo)
        ])

        # D. Entrenar y Predecir
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # E. Métricas
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        lista_recalls.append(rec)
        lista_precisions.append(prec)

    print("\n\n--- RESULTADOS FINALES ---")
    
    # Estadísticas
    metricas = {
        'Recall': lista_recalls,
        'Precision': lista_precisions
    }
    
    for nombre, lista in metricas.items():
        media = np.mean(lista)
        std = np.std(lista)
        print(f"\n{nombre}:")
        print(f"  Media: {media:.4f}")
        print(f"  Std:   {std:.4f}")
        print(f"  Min:   {np.min(lista):.4f}")
        print(f"  Max:   {np.max(lista):.4f}")

    # Veredicto
    std_foco = np.std(metricas[METRICA_FOCO])
    if std_foco <= UMBRAL_DESVIACION_ESTABLE:
        print(f"\n✅ VEREDICTO: ESTABLE (Std {std_foco:.4f} <= {UMBRAL_DESVIACION_ESTABLE})")
    else:
        print(f"\n⚠️ VEREDICTO: INESTABLE (Std {std_foco:.4f} > {UMBRAL_DESVIACION_ESTABLE})")

    # Visualización
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(lista_recalls, kde=True, ax=ax[0], color='blue').set_title('Recall')
    sns.histplot(lista_precisions, kde=True, ax=ax[1], color='green').set_title('Precision')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ejecutar_prueba_estabilidad()