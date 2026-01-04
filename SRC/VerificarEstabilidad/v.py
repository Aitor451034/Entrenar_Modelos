"""
Script ADAPTADO para la Verificación de Estabilidad (Balanced Random Forest + Pipeline Completo).

Este script ejecuta el pipeline exacto de tu modelo 'Sin Smote':
1. RobustScaler
2. PowerTransformer (Yeo-Johnson)
3. DropHighCorrelationFeatures (Filtro de colinealidad)
4. RFE (Recursive Feature Elimination con RandomForest)
5. BalancedRandomForestClassifier

Utiliza la extracción de características robusta (Savgol, derivadas físicas) 
y ejecuta el proceso múltiples veces con diferentes semillas para validar la estabilidad.
"""

# ==============================================================================
# 1. IMPORTACIONES DE BIBLIOTECAS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import warnings

# --- Funciones científicas y estadísticas (Del contexto original) ---
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis
from scipy.interpolate import PchipInterpolator

# --- Sklearn y métricas ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.base import BaseEstimator, TransformerMixin

# --- Imblearn ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import BalancedRandomForestClassifier

# Ignorar advertencias
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


# ==============================================================================
# 2. CONFIGURACIÓN DE LA PRUEBA DE ESTABILIDAD
# ==============================================================================

# --- ¡¡IMPORTANTE!! ---
# Pega aquí los mejores parámetros obtenidos en tu entrenamiento (Paso 3 del script original).
# Asegúrate de que las claves coincidan (model__..., selector__...).
MEJORES_PARAMETROS_GRID = {
    # Parámetros del Modelo (BalancedRandomForest)
    'model__class_weight': 'balanced_subsample',
    'model__max_depth': 8,              # Ajusta según tus resultados
    'model__max_features': 'log2',
    'model__min_samples_leaf': 17,
    'model__n_estimators': 221,
    
    # Parámetros del Selector (RFE)
    'selector__n_features_to_select': 9
}

# --- Configuración del Bucle ---
N_ITERACIONES_ESTABILIDAD = 50  # Número de semillas a probar
TEST_SIZE_RATIO = 0.3           # Mismo ratio que en el entrenamiento
METRICA_FOCO = 'Recall'         # 'Recall' o 'Precision'

# --- Umbrales ---
UMBRAL_DESVIACION_ESTABLE = 0.05 
UMBRAL_PERSONALIZADO = 0.3818  # El umbral óptimo que obtuviste en el Paso 5

# Ruta por defecto (opcional, si falla pedirá archivo manual)
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
# 3. CLASES Y FUNCIONES DE SOPORTE (Del Contexto Original)
# ==============================================================================

class DropHighCorrelationFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer que elimina características con una correlación absoluta mayor
    a un umbral especificado (colinealidad).
    """
    def __init__(self, threshold=0.95, feature_names=None):
        self.threshold = threshold
        self.feature_names = feature_names
        self.to_drop_indices_ = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_indices_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        return np.delete(X, self.to_drop_indices_, axis=1)

def leer_archivo(ruta_csv_defecto):
    print("Abriendo archivo ...")
    try:
        df = pd.read_csv(
            ruta_csv_defecto, encoding="utf-8", sep=";", on_bad_lines="skip", 
            header=None, quotechar='"', decimal=",", skiprows=3
        )
        print("¡Archivo CSV leído correctamente (Ruta por defecto)!")
        return df
    except Exception:
        print("No se encontró archivo por defecto o error de lectura. Seleccione manual.")
        root = tk.Tk(); root.withdraw()
        ruta = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not ruta: return None
        return pd.read_csv(
            ruta, encoding="utf-8", sep=";", on_bad_lines="skip", 
            header=None, quotechar='"', decimal=",", skiprows=3
        )

def preprocesar_dataframe_inicial(df):
    """Limpieza inicial idéntica al script de entrenamiento."""
    if df.shape[1] < 99: return None
    # Selección de columnas clave
    new_df = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]]
    new_df = new_df.iloc[:-2]
    new_df.columns = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "KAI2", "Ts2", "Fuerza", "Etiqueta datos"]
    
    for col in ["KAI2", "Ts2", "Fuerza"]:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    
    # Limpieza de duplicados
    new_df = new_df.drop_duplicates()
    new_df.index = range(1, len(new_df) + 1)
    
    # Conversión general
    for col in df.columns:
        if df[col].dtype == object:
            try: df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except: pass
    return new_df

def extraer_features_fila_por_fila(new_df):
    """
    Extracción ROBUSTA de características (Savgol, Derivadas, Física).
    Copiada del script de entrenamiento para asegurar consistencia.
    """
    X_calculado = []
    y_calculado = []
    
    print(f"Procesando {len(new_df)} puntos de soldadura (Algoritmo Robusto)...")

    for i in new_df.index:
        try:
            str_volt = new_df.loc[i, "Voltajes inst."]
            str_corr = new_df.loc[i, "Corrientes inst."]
            
            if pd.isna(str_volt) or pd.isna(str_corr): continue
                
            raw_volt = np.array([float(v) for v in str_volt.split(';') if v.strip()])
            raw_corr = np.array([float(v) for v in str_corr.split(';') if v.strip()])
            
            ns = int(new_df.loc[i, "Ns"])
            ts2 = int(new_df.loc[i, "Ts2"])
            t_soldadura = np.linspace(0, ts2, ns + 1)
            
            min_len = min(len(t_soldadura), len(raw_volt), len(raw_corr))
            if min_len < 10: continue 
            
            t_soldadura = t_soldadura[:min_len]
            raw_volt = raw_volt[:min_len]
            raw_corr = raw_corr[:min_len]

            # Recorte inicial (Ruido < 150A)
            start_idx = 0
            while start_idx < len(raw_corr) and raw_corr[start_idx] < 150:
                start_idx += 1
            
            if start_idx >= len(raw_volt): continue

            if start_idx > 0:
                raw_volt = raw_volt[start_idx:]
                raw_corr = raw_corr[start_idx:]
                t_soldadura = t_soldadura[start_idx:]
                t_soldadura = t_soldadura - t_soldadura[0] 
                ts2 = t_soldadura[-1]
                if len(raw_volt) <= 10: continue
            
            if len(np.unique(raw_corr)) <= 1 or len(np.unique(raw_volt)) <= 1: continue
            
            # Cálculo Resistencia y Suavizado
            valores_resistencia = np.divide(raw_volt, raw_corr, out=np.zeros_like(raw_volt), where=np.abs(raw_corr)>0.5)
            window = min(11, len(valores_resistencia) if len(valores_resistencia)%2!=0 else len(valores_resistencia)-1)
            if window > 3:
                r_smooth = savgol_filter(valores_resistencia, window_length=window, polyorder=3)
            else:
                r_smooth = valores_resistencia

            # Puntos Clave
            valles_temp, _ = find_peaks(-r_smooth)
            idx_valley_start = valles_temp[0] if len(valles_temp) > 0 else 0
            
            if idx_valley_start < len(r_smooth) - 1:
                idx_max_rel = np.argmax(r_smooth[idx_valley_start:])
                idx_max = idx_valley_start + idx_max_rel
            else:
                idx_max = np.argmax(r_smooth)

            idx_min = np.argmin(r_smooth)
            resistencia_max = r_smooth[idx_max]
            t_R_max = t_soldadura[idx_max]
            r_valley_start = r_smooth[idx_valley_start]
            t_valley_start = t_soldadura[idx_valley_start]
            r0 = r_smooth[0]
            r_e = r_smooth[-1]
            t_e = t_soldadura[-1]
            resistencia_min = np.min(r_smooth)
            t_min = t_soldadura[idx_min]

            # Energía
            i_amperios = raw_corr * 1000.0
            v_reales = raw_volt / 100.0
            t_segundos = t_soldadura / 1000.0
            potencia = v_reales * i_amperios
            # Usamos trapz para compatibilidad general
            q_joules = np.trapz(potencia, x=t_segundos)

            # Derivadas
            d1 = np.gradient(r_smooth, t_soldadura)
            d2 = np.gradient(d1, t_soldadura)
            d3 = np.gradient(d2, t_soldadura)
            
            max_curvatura = np.nanmax(np.abs(d2)) if not np.all(np.isnan(d2)) else np.nan
            max_jerk = np.nanmax(np.abs(d3)) if not np.all(np.isnan(d3)) else np.nan
            puntos_inflexion = np.sum(np.diff(np.sign(d2)) != 0)
            
            picos, _ = find_peaks(r_smooth)
            valles, _ = find_peaks(-r_smooth)
            num_picos = len(picos)
            num_valles = len(valles)

            # Pendientes y Estadísticas
            t_mean = np.mean(t_soldadura)
            r_mean = np.mean(r_smooth)
            numerador = np.sum((r_smooth - r_mean) * (t_soldadura - t_mean))
            denominador = np.sum((t_soldadura - t_mean)**2) 
            m_ols = numerador / denominador if denominador != 0 else np.nan

            idx_v_max = np.argmax(raw_volt)
            if idx_v_max < len(raw_volt) - 1:
                dt_v = t_soldadura[idx_v_max] - t_e
                pendiente_V = (raw_volt[idx_v_max] - raw_volt[-1]) / dt_v if dt_v != 0 else 0.0
            else:
                pendiente_V = 0.0

            dt_rise = t_R_max - t_valley_start
            k3 = (resistencia_max - r_valley_start) / dt_rise * 100 if dt_rise > 0 else 0.0
            
            delta_t_post = t_e - t_R_max
            k4 = ((r_e - resistencia_max) / delta_t_post * 100) if delta_t_post > 0 else 0.0

            pendientes_post = d1[idx_max:]
            num_negativas = np.sum(pendientes_post < 0)

            desv = np.std(r_smooth)
            rms = np.sqrt(np.mean(r_smooth**2))
            mediana = np.median(r_smooth)
            varianza = np.var(r_smooth)
            iqr = np.percentile(r_smooth, 75) - np.percentile(r_smooth, 25)
            asim = skew(r_smooth) if len(r_smooth) > 2 else np.nan
            curt = kurtosis(r_smooth) if len(r_smooth) > 2 else np.nan
            
            desv_pre_mitad_t = np.std(r_smooth[:len(r_smooth)//2])
            desv_R = np.std(r_smooth[:idx_max+1])
            r_mean_post_max = np.mean(r_smooth[idx_max:]) if idx_max < len(r_smooth) else np.nan

            area_total = np.trapz(r_smooth, t_soldadura)
            idx_mitad = len(t_soldadura) // 2
            area_pre_mitad = np.trapz(r_smooth[:idx_mitad], t_soldadura[:idx_mitad])
            area_post_mitad = area_total - area_pre_mitad
            
            rango_r_beta_alfa = resistencia_max - r0
            rango_r_e_beta = r_e - resistencia_max
            rango_t_e_beta = t_e - t_R_max
            rango_rmax_rmin = resistencia_max - resistencia_min
            rango_tiempo_max_min = t_R_max - t_min

            fila_features = [
                float(rango_r_beta_alfa), float(rango_t_e_beta), float(rango_r_e_beta), float(r0),
                float(k4), float(k3), float(iqr), float(desv_pre_mitad_t),
                float(r_e), float(desv), float(pendiente_V), float(rms),
                float(rango_rmax_rmin), float(r_mean_post_max), float(r_mean), float(desv_R),
                float(num_negativas), float(rango_tiempo_max_min), float(area_total), float(area_pre_mitad),
                float(area_post_mitad), float(max_curvatura), float(puntos_inflexion), float(max_jerk),
                float(mediana), float(varianza), float(asim), float(curt),
                float(num_picos), float(num_valles), float(q_joules), float(m_ols)
            ]
            
            # Manejo de Infinitos y NaNs
            fila_features = np.array(fila_features)
            fila_features[np.isinf(fila_features)] = np.nan
            
            # --- CAMBIO REALIZADO: ---
            # Comentamos o borramos la siguiente línea para que sea IDÉNTICO al entrenamiento.
            # Si en entrenamiento no imputaste con ceros, aquí tampoco debes hacerlo.
            # fila_features = np.nan_to_num(fila_features, nan=0.0) 
            
            X_calculado.append(fila_features)
            y_calculado.append(int(new_df.loc[i, "Etiqueta datos"]))

        except Exception:
            continue

    return np.array(X_calculado), np.array(y_calculado)

def cargar_datos_completos():
    df = leer_archivo(RUTA_CSV_POR_DEFECTO)
    if df is None: return None, None
    df_p = preprocesar_dataframe_inicial(df)
    if df_p is None: return None, None
    X_raw, y_raw = extraer_features_fila_por_fila(df_p)
    if len(X_raw) == 0: return None, None
    X = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
    y = pd.Series(y_raw, name="Etiqueta")
    return X, y

# ==============================================================================
# 4. FUNCIÓN PRINCIPAL DE VERIFICACIÓN DE ESTABILIDAD
# ==============================================================================

def ejecutar_prueba_estabilidad(n_iteraciones, umbral_std_estable, metric_focus):
    
    print("--- INICIANDO PRUEBA DE ESTABILIDAD (BALANCED RF + PIPELINE COMPLETO) ---")
    
    # 1. Cargar Datos
    X, y = cargar_datos_completos()
    if X is None: return

    # 2. Separar parámetros (Selector vs Modelo)
    n_features_rfe = MEJORES_PARAMETROS_GRID.get('selector__n_features_to_select', 15)
    
    # Extraemos los parámetros limpios para el BalancedRandomForest
    brf_params = {
        k.replace('model__', ''): v 
        for k, v in MEJORES_PARAMETROS_GRID.items() 
        if 'model__' in k
    }

    print(f"\nConfiguración:")
    print(f"- RFE Features a seleccionar: {n_features_rfe}")
    print(f"- Parámetros Balanced RF: {brf_params}")
    print(f"- Iteraciones: {n_iteraciones}")
    print(f"- Umbral de Decisión: {UMBRAL_PERSONALIZADO}\n")

    lista_recalls = []
    lista_precisions = []

    # 3. Bucle de Estabilidad
    for i in range(n_iteraciones):
        seed = i  # La semilla cambia en cada vuelta
        
        # A. Split (con seed variable)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE_RATIO, random_state=seed, stratify=y
        )
        
        num_defectos_test = np.sum(y_test)
        total_test = len(y_test)

        # B. Construir Pipeline EXACTO al de entrenamiento
        # 1. Scaler -> 2. Power -> 3. CorrFilter -> 4. RFE -> 5. Model
        
        selector_rfe = RFE(
            estimator=RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            n_features_to_select=n_features_rfe,
            step=1, # Paso del 10% aprox (o 1 feature si es int)
            verbose=0
        )
        
        modelo_brf = BalancedRandomForestClassifier(
            random_state=seed,
            sampling_strategy="auto",
            replacement=False,
            n_jobs=-1,
            **brf_params
        )

        pipeline = ImbPipeline([
            ('scaler', RobustScaler()),
            ('power', PowerTransformer(method='yeo-johnson', standardize=False)),
            ('corr_filter', DropHighCorrelationFeatures(threshold=0.95, feature_names=FEATURE_NAMES)),
            ('selector', selector_rfe),
            ('model', modelo_brf)
        ])

        # C. Entrenar
        pipeline.fit(X_train, y_train)
        
        # D. Predecir con UMBRAL PERSONALIZADO 
        y_probas = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_probas >= UMBRAL_PERSONALIZADO).astype(int)
        
        # E. Guardar métricas
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        lista_recalls.append(rec)
        lista_precisions.append(prec)
        print(f"Iteración {i+1:02d}: Defectos={num_defectos_test}/{total_test} | Recall={rec:.4f}, Precision={prec:.4f}")
    
    # 4. Resultados Estadísticos
    mean_rec, std_rec = np.mean(lista_recalls), np.std(lista_recalls)
    mean_prec, std_prec = np.mean(lista_precisions), np.std(lista_precisions)

    print("\n" + "="*40)
    print(" RESULTADOS ESTADÍSTICOS FINALES")
    print("="*40)
    print(f"RECALL    -> Media: {mean_rec:.4f} | Std: {std_rec:.4f} | Min: {np.min(lista_recalls):.4f} | Max: {np.max(lista_recalls):.4f}")
    print(f"PRECISION -> Media: {mean_prec:.4f} | Std: {std_prec:.4f} | Min: {np.min(lista_precisions):.4f} | Max: {np.max(lista_precisions):.4f}")
    
    # 5. Veredicto
    std_check = std_rec if metric_focus == 'Recall' else std_prec
    limite_aceptable = umbral_std_estable + 0.025
    print("\n--- VEREDICTO ---")
    if std_check <= umbral_std_estable:
        print(f"\n✅ [ ESTABLE ]")
        print(f"La desviación es excelente ({std_check:.4f} <= {umbral_std_estable}).")
        print("El modelo es robusto y fiable para producción.")
        
    elif umbral_std_estable < std_check <= limite_aceptable:
        print(f"\n⚠️ [ EN EL LÍMITE / ACEPTABLE ]")
        print(f"La desviación ({std_check:.4f}) supera el objetivo por poco.")
        print("-> CONCLUSIÓN: Es aceptable, pero vigilar en producción.")
        
    else:
        print(f"\n❌ [ INESTABLE ]")
        print(f"La desviación ({std_check:.4f}) es demasiado alta.")
        print("El modelo varía demasiado entre ejecuciones.")

    # 6. Gráficos
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(lista_recalls, kde=True, ax=ax[0], color='blue', bins=10)
    ax[0].set_title(f'Distribución Recall (Std={std_rec:.3f})')
    ax[0].axvline(mean_rec, c='r', ls='--')
    
    sns.histplot(lista_precisions, kde=True, ax=ax[1], color='green', bins=10)
    ax[1].set_title(f'Distribución Precision (Std={std_prec:.3f})')
    ax[1].axvline(mean_prec, c='r', ls='--')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ejecutar_prueba_estabilidad(N_ITERACIONES_ESTABILIDAD, UMBRAL_DESVIACION_ESTABLE, METRICA_FOCO)
# ==============================================================================