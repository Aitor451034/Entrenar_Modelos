"""
Script ACTUALIZADO para la Verificación de Estabilidad (Balanced Random Forest + RFE).

Este script ejecuta el pipeline exacto de tu entrenamiento:
1. StandardScaler
2. RFE (Recursive Feature Elimination)
3. BalancedRandomForestClassifier

Lo ejecuta múltiples veces con diferentes semillas ('random_state')
para asegurar que el RFE y el modelo son estables.
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
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import warnings

# Sklearn y métricas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier # Necesario para el RFE interno
from sklearn.metrics import recall_score, precision_score

# Imblearn (Para el Balanced Random Forest y el Pipeline)
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
# Pega aquí el diccionario de 'Mejores parámetros encontrados'
# EXACTAMENTE como sale de tu GridSearchCV.
# El script sabrá separar qué es para el Selector y qué para el Modelo.
MEJORES_PARAMETROS_GRID = {
    'model__class_weight': 'balanced',
    'model__max_depth': 6,
    'model__max_features': 'sqrt',
    'model__min_samples_leaf': 10,
    'model__n_estimators': 200,
    'selector__n_features_to_select': 15
}

# --- Configuración del Bucle ---
N_ITERACIONES_ESTABILIDAD = 50  # Número de repeticiones (seeds distintas)
TEST_SIZE_RATIO = 0.3           # Debe coincidir con tu entrenamiento
METRICA_FOCO = 'Recall'         # 'Recall' o 'Precision'

# --- Umbral ---
UMBRAL_DESVIACION_ESTABLE = 0.05 

# --- Poner Umbral dado por el modelo -----
UMBRAL_PERSONALIZADO = 0.3533

# ==============================================================================
# 3. FUNCIONES DE CARGA Y EXTRACCIÓN (LÓGICA ORIGINAL)
# ==============================================================================

RUTA_CSV_POR_DEFECTO = r"C:\Users\U5014554\Desktop\EntrenarModelo\DATA\Datos_Titanio25-26.csv"
FEATURE_NAMES = [
    "rango_r_beta_alfa", "rango_t_e_beta", "rango_r_e_beta", "resistencia_inicial", "k4", "k3",
    "rango_intercuartilico", "desv_pre_mitad_t", "resistencia_ultima", "desv", "pendiente_V",
    "rms", "rango_rmax_rmin", "r_mean_post_max", "r_mean", "desv_R_pre_max", "pendientes_negativas_post",
    "rango_tiempo_max_min", "area_bajo_curva", "area_pre_mitad", "area_post_mitad",
    "max_curvatura", "num_puntos_inflexion", "max_jerk", "mediana", "varianza", "asimetria",
    "curtosis", "num_picos", "num_valles", "q", "m_min_cuadrados"
]

def leer_archivo(ruta_csv_defecto):
    print("Abriendo archivo ...")
    try:
        df = pd.read_csv(ruta_csv_defecto, encoding="utf-8", sep=";", on_bad_lines="skip", header=None, quotechar='"', decimal=",", skiprows=3)
        print("¡Archivo CSV leído correctamente!")
        return df
    except FileNotFoundError:
        print("No se encontró archivo por defecto. Seleccione manual.")
        root = tk.Tk(); root.withdraw()
        ruta = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not ruta: return None
        return pd.read_csv(ruta, encoding="utf-8", sep=";", on_bad_lines="skip", header=None, quotechar='"', decimal=",")
    except Exception as e:
        print(f"Error: {e}")
        return None

def calcular_pendiente(resistencias, tiempos):
    if len(resistencias) <= 1: return [0]
    pendientes = []
    for i in range(len(resistencias) - 1):
        dt = tiempos[i+1] - tiempos[i]
        dr = resistencias[i+1] - resistencias[i]
        val = 0 if dt == 0 else (dr/dt)*100
        pendientes.append(round(np.nan_to_num(val, nan=0), 2))
    return pendientes

def calcular_derivadas(r, t):
    if len(r) <= 1: return np.array([0]), np.array([0]), np.array([0])
    d1 = np.gradient(r, t)
    d2 = np.gradient(d1, t)
    d3 = np.gradient(d2, t)
    return np.nan_to_num(d1), np.nan_to_num(d2), np.nan_to_num(d3)

def preprocesar_dataframe_inicial(df):
    new = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]].copy()
    new = new.iloc[:-2] # Eliminar filas vacías finales si existen
    new.columns = ["id", "Ns", "Corrientes", "Voltajes", "KAI2", "Ts2", "Fuerza", "Etiqueta"]
    for c in ["KAI2", "Ts2", "Fuerza"]: new[c] = pd.to_numeric(new[c], errors='coerce')
    for c in df.columns:
        if df[c].dtype == object: 
            try: df[c] = df[c].str.replace(',', '.').astype(float)
            except: pass
    new.index = range(1, len(new)+1)
    return new

def extraer_features_fila_por_fila(df):
    X_list, y_list = [], []
    print(f"Procesando {len(df)} filas con extracción COMPLETA...")
    
    for i in df.index:
        try:
            # --- 1. Parsing y Limpieza Básica ---
            v_str = df.loc[i, "Voltajes"]
            c_str = df.loc[i, "Corrientes"]
            if pd.isna(v_str) or pd.isna(c_str): continue
            
            # Convertir strings a listas de floats
            volts = [float(x) for x in v_str.split(';') if x.strip()]
            amps = [0.001 if float(x)==0 else float(x) for x in c_str.split(';') if x.strip()]
            
            # Calcular Resistencia (V/I)
            res = [round(v/c, 2) if c!=0 else 0 for v,c in zip(volts, amps)]
            res.append(0) # Padding final como en el original
            
            # Generar Vector de Tiempo
            ns, ts2 = int(df.loc[i, "Ns"]), int(df.loc[i, "Ts2"])
            t = np.linspace(0, ts2, ns+1).tolist()
            
            # Ajustar longitudes para evitar errores de índice
            m_len = min(len(t), len(res), len(volts), len(amps))
            t = t[:m_len]
            res = res[:m_len]
            volts = volts[:m_len]
            amps = amps[:m_len]
            
            if not t: continue

            # --- 2. Puntos Clave ---
            arr_res = np.array(res)
            rmax_idx = np.argmax(arr_res)
            rmax = arr_res[rmax_idx]
            t_rmax = t[rmax_idx]
            
            r0 = res[0]
            re = res[-2] if len(res)>1 else res[-1]
            te = t[-2] if len(t)>1 else t[-1]
            
            # --- 3. Cálculos Complejos Restaurados ---
            
            # A) Energía (q)
            watts = (np.array(volts)/100.0) * (np.array(amps)*10)
            q = np.trapz(watts, x=np.array(t)/1000.0)
            
            # B) Pendiente Voltaje (pendiente_V) - RESTAURADO
            v_max = max(volts)
            t_v_max = t[np.argmax(volts)]
            v_final = volts[-2] if len(volts)>1 else volts[-1]
            t_v_final = t[-2] if len(t)>1 else t[-1]
            delta_tv = t_v_max - t_v_final
            pendiente_V = 0 if delta_tv == 0 else (v_max - v_final) / delta_tv
            
            # C) Pendientes Negativas Post Max (pendientes_negativas_post) - RESTAURADO
            # Necesitamos calcular las pendientes punto a punto primero
            pendientes_raw = []
            for k in range(len(res) - 1):
                dt = t[k+1] - t[k]
                dr = res[k+1] - res[k]
                val = 0 if dt == 0 else (dr/dt)*100
                pendientes_raw.append(val)
            
            pendientes_post = pendientes_raw[rmax_idx:]
            pendientes_negativas_post = sum(1 for p in pendientes_post if p < 0)
            
            # D) Mínimos Cuadrados (m_min_cuadrados) - RESTAURADO
            t_mean = np.mean(t)
            r_mean_ols = np.mean(res)
            numerador = sum((r_mean_ols - ri) * (t_mean - ti) for ri, ti in zip(res, t))
            denominador = sum((r_mean_ols - ri) ** 2 for ri in res)
            m_min_cuadrados = 0 if denominador == 0 else (numerador / denominador)

            # E) Derivadas y Estadísticos estándar
            d1, d2, d3 = calcular_derivadas(res, t)
            
            # --- 4. Ensamblaje del Vector (Orden idéntico a FEATURE_NAMES) ---
            feats = [
                rmax - r0,                              # rango_r_beta_alfa
                te - t_rmax,                            # rango_t_e_beta
                re - rmax,                              # rango_r_e_beta
                r0,                                     # resistencia_inicial
                0 if (te-t_rmax)==0 else ((re-rmax)/(te-t_rmax))*100, # k4
                0 if (t_rmax-t[0])==0 else ((rmax-r0)/(t_rmax-t[0]))*100, # k3
                np.percentile(arr_res,75)-np.percentile(arr_res,25), # rango_intercuartilico
                np.std(res[:rmax_idx+1]) if rmax_idx>0 else 0,       # desv_pre_mitad_t
                re,                                     # resistencia_ultima
                np.std(res),                            # desv
                pendiente_V,                            # pendiente_V (YA NO ES 0)
                np.sqrt(np.mean(np.square(res))),       # rms
                rmax - np.min(res[:-1]) if len(res)>1 else 0, # rango_rmax_rmin
                np.mean(res[rmax_idx:]),                # r_mean_post_max
                np.mean(res[:-1]),                      # r_mean
                np.std(res[:rmax_idx]),                 # desv_R_pre_max
                pendientes_negativas_post,              # pendientes_negativas_post (YA NO ES 0)
                t_rmax - t[np.argmin(res[:-1])],        # rango_tiempo_max_min
                np.trapz(res, t),                       # area_bajo_curva
                np.trapz(res[:rmax_idx+1], t[:rmax_idx+1]), # area_pre_mitad
                np.trapz(res[rmax_idx:], t[rmax_idx:]),     # area_post_mitad
                np.max(np.abs(d2)),                     # max_curvatura
                len(np.where(np.diff(np.sign(d2)))[0]), # num_puntos_inflexion
                np.max(np.abs(d3)),                     # max_jerk
                np.median(res),                         # mediana
                np.var(res),                            # varianza
                skew(res),                              # asimetria
                kurtosis(res),                          # curtosis
                len(find_peaks(arr_res)[0]),            # num_picos
                len(find_peaks(-arr_res)[0]),           # num_valles
                q,                                      # q
                m_min_cuadrados                         # m_min_cuadrados (YA NO ES 0)
            ]
            
            # Limpieza final de NaNs e Infinitos
            feats = [float(np.nan_to_num(f, nan=0, posinf=0, neginf=0)) for f in feats]
            
            X_list.append(feats)
            y_list.append(int(df.loc[i, "Etiqueta"]))
            
        except Exception: 
            continue

    return np.array(X_list), np.array(y_list)

def cargar_datos_completos():
    df = leer_archivo(RUTA_CSV_POR_DEFECTO)
    if df is None: return None, None
    df_p = preprocesar_dataframe_inicial(df)
    X_raw, y_raw = extraer_features_fila_por_fila(df_p)
    if len(X_raw) == 0: return None, None
    X = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
    y = pd.Series(y_raw, name="Etiqueta")
    return X, y

# ==============================================================================
# 4. FUNCIÓN PRINCIPAL DE VERIFICACIÓN DE ESTABILIDAD
# ==============================================================================

def ejecutar_prueba_estabilidad(n_iteraciones, umbral_std_estable, metric_focus):
    
    print("--- INICIANDO PRUEBA DE ESTABILIDAD (BALANCED RF + RFE) ---")
    
    # 1. Cargar Datos
    X, y = cargar_datos_completos()
    if X is None: return

    # 2. Separar parámetros (Selector vs Modelo)
    # Extraemos cuántas features quiere el usuario para el RFE
    n_features_rfe = MEJORES_PARAMETROS_GRID.get('selector__n_features_to_select', 20)
    
    # Extraemos los parámetros limpios para el BalancedRandomForest
    brf_params = {
        k.replace('model__', ''): v 
        for k, v in MEJORES_PARAMETROS_GRID.items() 
        if 'model__' in k
    }

    print(f"\nConfiguración:")
    print(f"- RFE Features a seleccionar: {n_features_rfe}")
    print(f"- Parámetros Balanced RF: {brf_params}")
    print(f"- Iteraciones: {n_iteraciones}\n")

    lista_recalls = []
    lista_precisions = []

    # 3. Bucle de Estabilidad
    for i in range(n_iteraciones):
        seed = i  # La semilla cambia en cada vuelta
        
        # A. Split (con seed variable)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE_RATIO, random_state=seed, stratify=y
        )
        
        # --- NUEVAS LINEAS AQUI ---
        num_defectos_test = np.sum(y_test) # Cuenta cuantos 1 hay en y_test
        total_test = len(y_test)
        # --------------------------

        # B. Construir Pipeline EXACTO al de entrenamiento
        # Nota: El RFE necesita un estimador base. Usamos un RF estándar para seleccionar,
        # y luego el BalancedRF para clasificar, tal cual tu código original.
        
        selector_rfe = RFE(
            estimator=RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            n_features_to_select=n_features_rfe,
            step=0.1,
            verbose=0
        )
        
        modelo_brf = BalancedRandomForestClassifier(
            random_state=seed,  # Importante: seed variable
            sampling_strategy="auto",
            replacement=False,
            n_jobs=-1,
            **brf_params # Pasamos los params limpios (max_depth, etc.)
        )

        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('selector', selector_rfe),
            ('model', modelo_brf)
        ])

        # C. Entrenar
        pipeline.fit(X_train, y_train)
        
        # ---------------------------------------------------------
        # D. Predecir con UMBRAL PERSONALIZADO 
        # ---------------------------------------------------------
        # 1. Obtenemos la probabilidad de ser "Clase 1" (Defecto)
        y_probas = pipeline.predict_proba(X_test)[:, 1]
        
        # 2. Comparamos contra tu variable global
        y_pred = (y_probas >= UMBRAL_PERSONALIZADO).astype(int)
        # ---------------------------------------------------------
        
        # D. Guardar métricas
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
        # CASO 1: VERDE - ESTABLE
        print(f"\n✅ [ ESTABLE ]")
        print(f"La desviación es excelente ({std_check:.4f} <= {umbral_std_estable}).")
        print("El modelo es robusto y fiable para producción.")
        
    elif umbral_std_estable < std_check <= limite_aceptable:
        # CASO 2: AMARILLO - EN EL LÍMITE
        print(f"\n⚠️ [ EN EL LÍMITE / ACEPTABLE ]")
        print(f"La desviación ({std_check:.4f}) supera el objetivo por poco,")
        print(f"pero está dentro de la zona de tolerancia (+0.025).")
        print("-> CONCLUSIÓN: Es aceptable para una primera propuesta o fase piloto,")
        print("   pero se recomienda intentar conseguir más datos o reducir complejidad.")
        
    else:
        # CASO 3: ROJO - INESTABLE
        print(f"\n❌ [ INESTABLE ]")
        print(f"La desviación ({std_check:.4f}) supera el margen aceptable (> {limite_aceptable:.4f}).")
        print("El modelo varía demasiado. No es seguro confiar en él todavía.")
    # 6. Gráficos
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(lista_recalls, kde=True, ax=ax[0], color='blue', bins=8)
    ax[0].set_title(f'Distribución Recall (Std={std_rec:.3f})')
    ax[0].axvline(mean_rec, c='r', ls='--')
    
    sns.histplot(lista_precisions, kde=True, ax=ax[1], color='green', bins=8)
    ax[1].set_title(f'Distribución Precision (Std={std_prec:.3f})')
    ax[1].axvline(mean_prec, c='r', ls='--')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ejecutar_prueba_estabilidad(N_ITERACIONES_ESTABILIDAD, UMBRAL_DESVIACION_ESTABLE, METRICA_FOCO)