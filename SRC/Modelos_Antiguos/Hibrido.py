import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc, fbeta_score, make_scorer, classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
from sklearn import metrics
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV # Usaremos esto en lugar de GridSearchCV
from scipy.stats import randint                     # Para la búsqueda de parámetros
from sklearn.model_selection import GridSearchCV  # Reemplaza a RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier # Modelo alternativo
from imblearn.pipeline import Pipeline as ImbPipeline 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict

def calcular_parametros():
    """Return the array of 32 features and the labels of each spot weld.
    
    Returns:
        X_calculado (numpy.ndarray):   2D array of shape (n_samples, 32) containing features
        y_calculado (numpy.ndarray):   1D array of shape (n_samples,) containing labels (0: No defect, 1: Defect)
    """
# ==============================================================================
    def leer_archivo():
        """Read a CSV file containing spot weld data and return it as a DataFrame.
        
        If the default file path is not found, a file dialog will prompt the user to select the file manually

        Returns:
            df (pandas.Dataframe): DataFrame containing the contents of the CSV file
        """
        print("Abriendo archivo ...")
        ruta_csv = r"C:\Users\U5014554\Desktop\EntrenarModelo\DATA\Inputs_modelo_pegado_con_datos3.csv"
        try:
            df = pd.read_csv(ruta_csv, encoding = "utf-8", sep = ";", on_bad_lines = "skip", header = None, quotechar = '"', decimal = ",", skiprows = 3)
            print("¡Archivo CSV leído correctamente!")
            return df
        except FileNotFoundError:
            print("No se ha encontrado el archivo ...")
            root = tk.Tk()
            root.withdraw()  
            ruta_csv = filedialog.askopenfilename(title="Seleccionar archivo que contiene los datos",
                                            filetypes=[("Archivos de CSV", "*.csv")])
            df = pd.read_csv(ruta_csv, encoding = "utf-8", sep = ";", on_bad_lines= "skip", header = None, quotechar = '"', decimal = ",")
            return df
        except Exception as e:
            print("Se produjo un error al leer el archivo Excel:", e)
# ==============================================================================
    def calcular_pendiente(resistencias, tiempos):
        """Calculate the slope (rate of change) between two consecutive values of resistance and time values.

        Arguments:
            resistencias (list): Sequence of resistance values
            tiempos (list): Sequence of time values corresponding to each resistance value

        Returns:
            pendientes (list): Sequence of slope values
        """
        if len(resistencias) <= 1 or len(tiempos) <= 1:
            return [0]  
        pendientes = []
        # Recorrer los elementos de las listas, excepto el último
        for i in range(len(resistencias) - 1):
            # Calcular la pendiente entre el punto actual y el anterior
            pendiente_actual = np.nan_to_num((((resistencias[i + 1] - resistencias[i]) / (tiempos[i + 1] - tiempos[i]))*100), nan=0)
            # Añadir la pendiente a la lista
            pendientes.append(round(pendiente_actual,2))
        return pendientes
# ==============================================================================    
    def calcular_derivadas(resistencias, tiempos):
        """Calculate the first (slope), second (curvature) and third (rate of change of curvature) derivatives of a resistance-time curve.

        Arguments:
            resistencias (list): Sequence of resistance values
            tiempos (list): Sequence of time values corresponding to each resistance value

        Returns:
            primera_derivada (numpy.ndarray): First derivative (slope) of the resistance-time curve
            segunda_derivada (numpy.ndarray): Second derivative (curvature) of the resistance-time curve
            tercera_derivada (numpy.ndarray): Third derivative (rate of change of curvature) of the resistance-time curve
        """
        if len(resistencias) <= 1 or len(tiempos) <= 1:
            return [0], [0], [0]  # Retorna listas con un solo elemento cero si hay datos insuficientes
        primera_derivada = np.nan_to_num(np.gradient(resistencias, tiempos), nan=0)         # Pendiente de la curva
        segunda_derivada = np.nan_to_num(np.gradient(primera_derivada, tiempos), nan=0)     # Forma de la curva (concava/convexa)
        tercera_derivada = np.nan_to_num(np.gradient(segunda_derivada, tiempos), nan=0)     # Oscilaciones de la curva
        return primera_derivada, segunda_derivada, tercera_derivada  
# ==============================================================================
    df = leer_archivo()
    new_df = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]]  
    new_df = new_df.iloc[: -2]
    new_df.columns = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "KAI2", "Ts2", "Fuerza", "Etiqueta datos"] 
    for col in ["KAI2", "Ts2", "Fuerza"]: 
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    new_df = new_df.round({col: 4 for col in new_df.select_dtypes(include='float64').columns})
    new_df.index = range(1, len(new_df) + 1)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except:
                pass
# ==============================================================================
    X_calculado = []
    y_calculado = []
    # Por cada fila de datos (1 punto de soldadura) se leen datos y se realizan transformaciones que se almacenan en X
    for i in new_df.index:
        # Valores de voltaje instantaneo
        datos_voltaje = new_df.loc[i, "Voltajes inst."]
        if pd.isna(datos_voltaje):
            break                                              
        valores_voltaje = [float(valor) for valor in datos_voltaje.split(';') if valor.strip()]
        valores_voltaje = [round(valor, 0) for valor in valores_voltaje]
        datos_corriente = new_df.loc[i, "Corrientes inst."]
        valores_corriente = [0.001 if float(valor) == 0 else float(valor) for valor in datos_corriente.split(';') if valor.strip()]
        valores_corriente = [round(valor, 0) for valor in valores_corriente]
        valores_resistencia = [v / c if c != 0 else 0 for v, c in zip(valores_voltaje, valores_corriente)]
        valores_resistencia = [round(valor, 2) for valor in valores_resistencia]
        valores_resistencia.append(0)   
        # ID de cada punto
        id_punto = new_df.loc[i, "id punto"]
        # Nº de datos por punto
        ns = int(new_df.loc[i, "Ns"])
        # Tiempo total de soldadura
        ts2 = int(new_df.loc[i,"Ts2"])
        # Vector que contiene el tiempo de soldadura
        t_soldadura = ((np.linspace(0, ts2, ns+1))).tolist()
        # Resistencia máxima
        resistencia_max = max(valores_resistencia)                              
        # Indice de ns que indica el mayor valor de resistencia
        I_R_max = np.argmax(valores_resistencia)
        # Tiempo en el que se alcanza la resistencia máxima
        t_R_max = int(t_soldadura[I_R_max])                                 
        # Cáculo de la pendiente desde la resistencia inicial hasta la resistencia final (0)
        r_final = valores_resistencia[-1]
        t_final = t_soldadura[-1]
        r0 = valores_resistencia[0]
        t0 = t_soldadura[0]
        pendiente_total = np.nan_to_num((((r_final-r0)/(t_final)*100)), nan=0)          
        # Intensidad de soldadura 
        kAI2 = new_df.loc[i, "KAI2"]
        # Fuerza de soldadura
        f = new_df.loc[i, "Fuerza"]
        # Calor total generado
        q = np.nan_to_num(((((kAI2 * 1000.0) ** 2) * (ts2 / 1000.0)) / (f * 10.0)), nan=0)
        # Área bajo la curva de resistencia dinámica
        area_bajo_curva = np.nan_to_num(np.trapz(valores_resistencia, t_soldadura), nan=0)
        # Valor de la última resistencia antes del valor 0
        resistencia_ultima = valores_resistencia[-2]
        # Pendiente k4: desde resistencia máxima hasta última resistencia antes de 0
        r_e = valores_resistencia[-2]
        t_e = t_soldadura[-2]
        try:
            if t_e != t0:
                k4 = ((r_e - resistencia_max) / (t_e - t_R_max)) * 100
            else:
                k4 = 0  # Cuando t_soldadura_max == t0, asignar k4 a 0
        except ZeroDivisionError:
            k4 = 0
        k4 = np.nan_to_num(k4, nan=0)
        # Desviación estandar de todos los valores de resistencia
        desv = np.nan_to_num(np.std(valores_resistencia), nan=0)
        # Valor RMS
        rms = np.nan_to_num(np.sqrt(np.mean(np.square(valores_resistencia))), nan=0)
        # Intervalo de tiempo entre el tiempo en el que sucede la resistencia máxima y mínima
        resistencia_min = min(valores_resistencia[:-1])     # valor de resistencia mínima
        t_min = np.argmin(valores_resistencia[:-1])         # índice de la resistencia mínima 
        t_soldadura_min = t_soldadura[t_min]                # tiempo en el que ocurre la resistencia mínima 
        rango_tiempo_max_min = np.nan_to_num(t_R_max - t_soldadura_min, nan=0)
        # Intervalo entre resistencia máxima y resistencia mínima
        rango_rmax_rmin = np.nan_to_num(resistencia_max - resistencia_min, nan=0)
        # Pendiente de la tensión instantanea entre la tensión máxima y la tensión última
        voltaje_max = max(valores_voltaje)  
        t_max_v = np.argmax(valores_voltaje)
        t_voltaje_max = t_soldadura[t_max_v]    # Tiempo tensión máxima
        voltaje_final = valores_voltaje[-2]  
        t_voltaje_final = t_soldadura[-2]       # Tiempo tensión última
        # --- INICIO DE LA CORRECCIÓN ---
        try:
            pendiente_V = ((voltaje_max - voltaje_final)/(t_voltaje_max - t_voltaje_final))
        except ZeroDivisionError:
            pendiente_V = 0
        pendiente_V = np.nan_to_num(pendiente_V, nan=0)
        # --- FIN DE LA CORRECCIÓN ---
        # Valor medio de resistencia dinámica después de la resistencia máxima
        r_mean_post_max = np.nan_to_num(np.mean(valores_resistencia[t_R_max:]), nan=0)
        # Valor de la resistencia inicial	
        resistencia_inicial = np.nan_to_num(r0, nan = 2000)
        # Pendiente k3: desde resistencia inicial hasta resistencia máxima	
        try:
            if t_R_max != t0:
                k3 = ((resistencia_max - r0) / (t_R_max - t0)) * 100
            else:
                k3 = 0  # Cuando t_soldadura_max == t0, asignar k3 a 0
        except ZeroDivisionError:
            k3 = 0
        k3 = np.nan_to_num(k3, nan=0)
        # Valor medio de la resistencia dinámica en todo el intervalo de tiempo
        r_mean = np.nan_to_num(np.mean(valores_resistencia[:-1]), nan=0)
        # Intervalo (diferencia) entre el valor de la resistencia máxima y la resistencia inicial
        rango_r_beta_alfa = np.nan_to_num(resistencia_max - r0, nan=0)
        # Intervalo (diferencia) entre el valor de la resistencia máxima y la resistencia final antes de 0	
        rango_r_e_beta = np.nan_to_num(r_e - resistencia_max, nan=0)
        # Intervalo (diferencia) entre el instante de tiempo en el que se da la resistencia máxima y en el que se da la resistencia última antes de 0  
        rango_t_e_beta = np.nan_to_num(t_e - t_R_max, nan=0)
        # Desviación estandar de resistencia dinámica antes del valor de la resistencia máxima
        desv_R = np.nan_to_num(np.std(valores_resistencia[:t_R_max]), nan=0)        
        # Nº de pendientes negativas despues de alcanzar el valor de resistencia máxima	
        pendientes = calcular_pendiente(valores_resistencia, t_soldadura)
        pendientes_post_max = pendientes[(t_R_max + 1):]
        pendientes_pre_max = pendientes[:t_R_max ]
        pendientes_negativas_post = sum(1 for p in pendientes_post_max if p < 0)
        # Area bajo la curva de resistencias dinámicas antes de alcanzar el valor de la resitencia máxima
        valores_resistencia_hasta_R_max = [valores_resistencia[j] for j in range(len(t_soldadura)) if t_soldadura[j] <= t_R_max]
        valores_tiempo_hasta_R_max = [t_soldadura[j] for j in range(len(t_soldadura)) if t_soldadura[j] <= t_R_max]
        area_pre_mitad = np.nan_to_num(np.trapz(valores_resistencia_hasta_R_max, valores_tiempo_hasta_R_max), nan=0)
        # Area bajo la curva de resistencias dinámicas despues de alcanzar el valor de la resitencia máxima
        valores_resistencia_desde_R_max = [valores_resistencia[j] for j in range(len(t_soldadura)) if t_soldadura[j] >= t_R_max]
        valores_tiempo_desde_R_max = [t_soldadura[j] for j in range(len(t_soldadura)) if t_soldadura[j] >= t_R_max]
        area_post_mitad = np.nan_to_num(np.trapz(valores_resistencia_desde_R_max, valores_tiempo_desde_R_max), nan=0)	
        # Desviación estandar de la curva de resistencia dinámica antes de alcanzar el valor máximo de resistencia
        try:
            desv_pre_mitad_t = np.nan_to_num(np.std(valores_resistencia_hasta_R_max), nan=0)
        except ValueError:
            desv_pre_mitad_t = 0
        # Calcular la primera, segunda y tercera derivada	
        primera_derivada, segunda_derivada, tercera_derivada = calcular_derivadas(valores_resistencia, t_soldadura)
        # Hallar la curvatura máxima
        try:
            max_curvatura = np.nan_to_num(np.max(np.abs(segunda_derivada)), nan=0)
        except ValueError:
            max_curvatura = 0
        # Nº de puntos de inflexión de la curva de resistencia dinámica
        try:
            puntos_inflexion = np.where(np.diff(np.sign(segunda_derivada)))[0]  
            num_puntos_inflexion = np.nan_to_num(len(puntos_inflexion), nan=0)
        except ValueError:
            num_puntos_inflexion = 0
        # Máximo valor absoluto de la tercera derivada
        try:
            max_jerk = np.nan_to_num(np.max(np.abs(tercera_derivada)), nan=0)
        except ValueError:
            max_jerk = 0
        # Mediana de los valores de resistencia dinámica
        try:
            mediana = np.nan_to_num(np.median(valores_resistencia), nan=0)
        except ValueError:
            mediana = 0    
        # Varianza de los valores de resistencia dinámica
        try:
            varianza = np.nan_to_num(np.var(valores_resistencia), nan=0)
        except ValueError:
            varianza = 0 
        # Rango intercuartílico
        try:
            rango_intercuartilico = np.nan_to_num((np.percentile(valores_resistencia, 75) - np.percentile(valores_resistencia, 25)), nan=0)
        except ValueError:
            varianza = 0 
        # Asimetria de la curva de resistencia dinámica
        try:    
            asimetria = np.nan_to_num(skew(valores_resistencia), nan=0)
        except ValueError:
            asimetria = 0 
        # Curtosis de la curva de resistencia dinámica
        try:
            curtosis = np.nan_to_num(kurtosis(valores_resistencia), nan=0)
        except ValueError:
            curtosis = 0
        # Nº de picos y valles de la curva de resistencia dinámica
        valores_resistencia_np = np.array(valores_resistencia)
        picos, propiedades_picos = find_peaks(valores_resistencia_np, height=0)
        valles, propiedades_valles = find_peaks(-valores_resistencia_np)
        num_picos = np.nan_to_num(len(picos), nan=0)        # Número de picos
        num_valles = np.nan_to_num(len(valles), nan=0)      # Número de valles
        # Pendiente de la recta de mínimos cuadrados.
        t_mean = np.nan_to_num(np.mean(t_soldadura), nan = 0)
        numerador = sum((r_mean - ri)*(t_mean- ti) for ri, ti in zip(valores_resistencia, t_soldadura))
        denominador = sum((r_mean - ri) ** 2 for ri in valores_resistencia)
        m_min_cuadrados = []
        if denominador == 0:
            m_min_cuadrados = (0)
        else:
            m_min_cuadrados = (numerador/denominador)       
        # Se actualiza el valor del conjunto que contiene los datos transformados. Estos serán los inputs del modelo (32 parámetros)
        X_calculado.append([
            float(rango_r_beta_alfa),           #Var. predictora 0
            float(rango_t_e_beta),              #Var. predictora 1
            float(rango_r_e_beta),              #Var. predictora 2
            float(resistencia_inicial),         #Var. predictora 3
            float(k4),                          #Var. predictora 4
            float(k3),                          #Var. predictora 5
            float(rango_intercuartilico),       #Var. predictora 6
            float(desv_pre_mitad_t),            #Var. predictora 7
            float(resistencia_ultima),          #Var. predictora 8
            float(desv),                        #Var. predictora 9
            float(pendiente_V),                 #Var. predictora 10
            float(rms),                         #Var. predictora 11
            float(rango_rmax_rmin),             #Var. predictora 12
            float(r_mean_post_max),             #Var. predictora 13
            float(r_mean),                      #Var. predictora 14
            float(desv_R),                      #Var. predictora 15
            float(pendientes_negativas_post),   #Var. predictora 16
            float(rango_tiempo_max_min),        #Var. predictora 17
            float(area_bajo_curva),             #Var. predictora 18
            float(area_pre_mitad),              #Var. predictora 19
            float(area_post_mitad),             #Var. predictora 20
            float(max_curvatura),               #Var. predictora 21
            float(num_puntos_inflexion),        #Var. predictora 22
            float(max_jerk),                    #Var. predictora 23
            float(mediana),                     #Var. predictora 24
            float(varianza),                    #Var. predictora 25
            float(asimetria),                   #Var. predictora 26
            float(curtosis),                    #Var. predictora 27
            float(num_picos),                   #Var. predictora 28
            float(num_valles),                  #Var. predictora 29
            float(q),                           #Var. predictora 30
            float(m_min_cuadrados)])            #Var. predictora 31
    X_calculado = np.array(X_calculado)
    # Conjunto que contiene las etiquetas de los puntos OK (0) / No OK (1)
    for i in new_df.index:
        valor_i = int(new_df.loc[i, "Etiqueta datos"])
        y_calculado.append(valor_i)      
    y_calculado = np.array([int(v) for v in y_calculado])
    return X_calculado, y_calculado
# ==============================================================================
X, y = calcular_parametros()
print(type(X))
print(type(y))
# Preparar los datos
X = pd.DataFrame(X)
X = X.applymap(lambda x: round(x, 4))
print(X)
y = pd.Series(y)

# ==============================================================================
# 1. Separar en entrenamiento y validación (Tu código)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Guardamos los nombres de las columnas para más tarde
train_cols = X_train.columns
test_cols = X_test.columns

# ==============================================================================
# 2. ¡NUEVO! Escalar los datos (Soluciona problemas de convergencia)
# ==============================================================================
# La Regresión Logística L1 es muy sensible a la escala.
# Esto es fundamental para que el modelo converja rápida y correctamente.
print("Escalando datos...")
scaler = StandardScaler()

# Ajustamos el scaler SÓLO con X_train y transformamos X_train
X_train_scaled = scaler.fit_transform(X_train)

# Usamos el scaler YA AJUSTADO para transformar X_test
X_test_scaled = scaler.transform(X_test)

# Convertimos de nuevo a DataFrame para que el resto de tu código funcione
X_train = pd.DataFrame(X_train_scaled, columns=train_cols)
X_test = pd.DataFrame(X_test_scaled, columns=test_cols)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
print("Datos escalados.")
# Contamos cuántos 0 y cuántos 1 hay EN EL SET DE ENTRENAMIENTO
conteo_clases = y_train.value_counts()
ratio_desbalanceo = conteo_clases[0] / conteo_clases[1]

print(f"Ratio de desbalanceo (0s / 1s): {ratio_desbalanceo}")

# ==============================================================================
# 3. ¡MODIFICADO! Búsqueda de Hiperparámetros para HÍBRIDO (SMOTE + RandomForest)
# ==============================================================================
print("Iniciando búsqueda de hiperparámetros para SMOTE + RandomForest...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definir el scorer para F2-score (Tu código, sin cambios)
fbeta = 2
f2_scorer = make_scorer(fbeta_score, beta=fbeta)

# --- 1. Definir el modelo (CAMBIO) ---
# Ahora usamos RandomForestClassifier. 
# No necesitamos 'scale_pos_weight' o 'eval_metric'.
# Omitimos 'class_weight' para dejar que SMOTE haga todo el trabajo.
modelo_hibrido_rf = RandomForestClassifier(random_state=42)

# --- 2. Definir el Pipeline (CAMBIO) ---
# El pipeline ahora usa 'modelo_hibrido_rf'
pipeline_rf = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', modelo_hibrido_rf)
])

# --- 3. Definir el GRID de parámetros (CAMBIO) ---
# Usamos parámetros de RandomForest, con el prefijo 'model__'
# Este grid está optimizado para GENERALIZAR (evitar overfitting)
param_grid_rf = {
    # Número de árboles
    'model__n_estimators': [200, 300, 400],
    
    # Profundidad (¡CLAVE! Se quita 'None' para evitar memorización)
    'model__max_depth': [5, 7, 10],
    
    # Mínimo de muestras por hoja (¡CLAVE! > 1 fuerza a generalizar)
    'model__min_samples_leaf': [3, 5, 10],
    
    # Mínimo de muestras para dividir
    'model__min_samples_split': [2, 5, 10],
    
    # Características por árbol (previene que todos los árboles sean iguales)
    'model__max_features': ['sqrt', 'log2']
}

# --- 4. Configurar la Búsqueda (CAMBIO) ---
# Apuntamos a los nuevos 'pipeline_rf' y 'param_grid_rf'
search_cv = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid_rf,
    cv=skf,
    scoring=f2_scorer,       # Sigues optimizando para F2-Score
    n_jobs=-1,
    verbose=2
)

# --- 5. Entrenar (Sin cambios) ---
print("Entrenando GridSearchCV con SMOTE + RandomForest... (Esto puede tardar)")
search_cv.fit(X_train, y_train)

# --- 6. Obtener el mejor modelo (Sin cambios) ---
mejor_modelo = search_cv.best_estimator_
print("Entrenamiento de SMOTE + RandomForest completado.")

# ==============================================================================
# 4. Resultados del modelo (Ajustado a RandomizedSearchCV)
# ==============================================================================
print(f"Mejores parámetros encontrados: {search_cv.best_params_}")

# El mejor F2-score encontrado durante la búsqueda
mejor_score = search_cv.best_score_
print(f"Mejor score F2 (en CV): {mejor_score}")

# ==============================================================================
# 5. Importancia de las características
# ==============================================================================
# Random Forest no tiene .coef_, tiene .feature_importances_
importancias = mejor_modelo.named_steps['model'].feature_importances_
df_importancias = pd.DataFrame({
    'predictor': X_train.columns,
    'importancia': importancias
}).sort_values(by='importancia', ascending=True)

# ==============================================================================
# GRAFICAR LOS 32 FEATURES DE IMPORTANCIA
# ==============================================================================
print("Importancia de las 32 características (features) para el modelo XGBoost:")
print(df_importancias)
fig, ax = plt.subplots(figsize=(10, 8)) # Más alto para 32 características
ax.barh(df_importancias.predictor, df_importancias.importancia)
plt.yticks(size=8)
ax.set_xlabel('Importancia de la Característica')
ax.set_ylabel('Variable Predictora')
ax.set_title('Importancia de Características del Random Forest')
plt.tight_layout()
plt.show()
# ==============================================================================
# Realizar las predicciones en el conjunto de prueba
predicciones1 = mejor_modelo.predict(X_test)
y_pred_proba = mejor_modelo.predict_proba(X_test)
# ==============================================================================
# Calcular la matriz de confusión sin optimizar umbral
matriz_confusion = confusion_matrix(y_test, predicciones1)
nombres_etiquetas = ["Punto SIN DEFECTO","Punto CON PEGADO"]
fig, ax = plt.subplots()
tick_marks = np.arange(len(nombres_etiquetas))
plt.xticks(tick_marks, nombres_etiquetas)
plt.yticks(tick_marks, nombres_etiquetas)
sns.heatmap(
    pd.DataFrame(matriz_confusion),
    annot=True,
    cmap="YlGnBu",
    fmt='g',
    cbar = False,
    xticklabels = ["Predicho Sin defecto", "Predicho Pegado"], yticklabels = ["Real Sin defecto", "Real Pegado"])
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title("Matriz de Confusión XGBOOST sin optimizar (umbral = 0.5)")
plt.ylabel('Etiqueta real')
plt.xlabel('Predicción')
plt.show()
# ==============================================================================
# 7.Optimizar para "Sinergia": MAX(Recall) sujeto a MIN(Precision)
# ==============================================================================
print("Optimizando umbral para 'Sinergia': MAX(Recall) sujeto a Precision >= 0.70...")

# 1. Usar cross_val_predict para obtener predicciones "limpias" (fuera de fold)
#    (Esto es de la corrección anterior y es el método robusto)
print("Obteniendo predicciones de validación cruzada (puede tardar)...")
y_probas_cv = cross_val_predict(
    mejor_modelo, 
    X_train, 
    y_train, 
    cv=skf, 
    method='predict_proba', 
    n_jobs=-1
)[:, 1] # Obtenemos solo la probabilidad de la clase "1"

# 2. Ahora, buscar el mejor umbral que cumpla tu REGLA DE NEGOCIO
lista_umbrales = np.linspace(0.01, 0.99, 1000)

#DEFINE AQUÍ TU PRECISIÓN MÍNIMA ACEPTABLE
PRECISION_MINIMA = 0.6 

best_recall = -1
optimal_threshold = 0.01
best_precision_at_best_recall = 0

for thresh in lista_umbrales:
    y_pred_thresh = np.where(y_probas_cv >= thresh, 1, 0)
    
    prec = precision_score(y_train, y_pred_thresh, zero_division=0)
    rec = recall_score(y_train, y_pred_thresh, zero_division=0)

    # 1. ¿Cumple la condición de precisión?
    if prec >= PRECISION_MINIMA:
        # 2. Si la cumple, ¿es el mejor RECALL que hemos visto?
        if rec > best_recall:
            best_recall = rec
            optimal_threshold = thresh
            best_precision_at_best_recall = prec
        # 3. Si empata en Recall, preferimos el umbral más bajo
        #    (más sensible) que AÚN CUMPLE la precisión
        elif rec == best_recall:
             optimal_threshold = min(optimal_threshold, thresh)

# --- Comprobación y Fallback ---
if best_recall == -1:
    print(f"¡ADVERTENCIA!No se encontró NINGÚN umbral que cumpla 'Precision >= {PRECISION_MINIMA}'.")
    print("El modelo no puede satisfacer este requisito. Intenta bajar la PRECISION_MINIMA.")
    optimal_threshold = 0.5 # Fallback simple
else:
    print("¡Éxito! Se encontró un umbral que cumple la 'Sinergia'.")
    print(f"   -> Umbral óptimo: {optimal_threshold:.4f}")
    print(f"   -> Recall resultante (en CV): {best_recall:.4f}")
    print(f"   -> Precision resultante (en CV): {best_precision_at_best_recall:.4f}")
# ==============================================================================
# 8.Calcular la matriz de confusión con el umbral óptimo
# ============================================================================== 
predicciones_test_proba = mejor_modelo.predict_proba(X_test)[:, 1]
predicciones_test_binarias = np.where(predicciones_test_proba >= optimal_threshold, 1, 0)
matriz_confusion_opt = confusion_matrix(y_test, predicciones_test_binarias)
fig, ax = plt.subplots()
plt.xticks(tick_marks, nombres_etiquetas)
plt.yticks(tick_marks, nombres_etiquetas)
sns.heatmap(
    pd.DataFrame(matriz_confusion_opt),
    annot=True,
    cmap="YlGnBu",
    fmt='g',
    cbar=False,
    xticklabels = ["Predicho Sin defecto", "Predicho Pegado"], yticklabels = ["Real Sin defecto", "Real Pegado"])
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title(f"Matriz de Confusión XGBOOST con Umbral Óptimo = {optimal_threshold:.4f}")
plt.ylabel('Etiqueta real')
plt.xlabel('Predicción')
plt.show()
# ==============================================================================
# Evaluar el modelo utilizando classification_report para ver la exactitud, la precisión y la recuperación con el umbral óptimo
target_names = ['sin defecto', 'con defecto']
print(classification_report(y_test, predicciones_test_binarias, target_names=target_names))
# ==============================================================================
# Curva ROC con el umbral óptimo
fpr1, tpr1, _ = metrics.roc_curve(y_test,  predicciones_test_binarias)
auc = metrics.roc_auc_score(y_test, predicciones_test_binarias)
plt.plot(fpr1,tpr1,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title(f"Curva ROC V1")
plt.show()

# ==============================================================================
# 9. ANÁLISIS DE ERRORES (FALSOS NEGATIVOS Y FALSOS POSITIVOS)
# ==============================================================================
print("\n--- INICIANDO ANÁLISIS DE ERRORES EN EL TEST SET ---")

# 1. Ya tenemos las predicciones del Test Set del paso anterior
# predicciones_test_proba = mejor_modelo.predict_proba(X_test)[:, 1]
# predicciones_test_binarias = np.where(predicciones_test_proba >= optimal_threshold, 1, 0)
# y_test = (el y_test de tu train_test_split)
# X_test = (el X_test de tu train_test_split)

# 2. Creemos un DataFrame de análisis para verlo todo junto.
# Usamos X_test.copy() para no modificar el original.
df_analisis = X_test.copy()
df_analisis['Etiqueta_Real'] = y_test
df_analisis['Probabilidad_Defecto'] = predicciones_test_proba
df_analisis['Prediccion_Binaria'] = predicciones_test_binarias

# 3. Filtrar para encontrar FALSOS NEGATIVOS (FN)
# La Etiqueta Real era 1, pero el modelo predijo 0
condicion_FN = (df_analisis['Etiqueta_Real'] == 1) & (df_analisis['Prediccion_Binaria'] == 0)
falsos_negativos = df_analisis[condicion_FN]

print(f"\n[INFORME] Se han encontrado {len(falsos_negativos)} Falsos Negativos (Defectos no detectados):")
print(falsos_negativos)


# 4. Filtrar para encontrar FALSOS POSITIVOS (FP)
# La Etiqueta Real era 0, pero el modelo predijo 1
condicion_FP = (df_analisis['Etiqueta_Real'] == 0) & (df_analisis['Prediccion_Binaria'] == 1)
falsos_positivos = df_analisis[condicion_FP]

print(f"\n[INFORME] Se han encontrado {len(falsos_positivos)} Falsos Positivos (Falsas Alarmas):")
print(falsos_positivos)

# ==============================================================================
# 10.Guardar modelo, umbral Y SCALER
# ==============================================================================
# Es VITAL guardar el 'scaler' junto con el modelo,
# para poder escalar los datos nuevos exactamente igual.
print("Guardando modelo, scaler y umbral en 'modelo_con_umbral.pkl'...")
modelo_y_umbral = {
    "modelo": mejor_modelo,
    "scaler": scaler,               # <-- ¡NUEVO Y MUY IMPORTANTE!
    "umbral de predicciones": optimal_threshold
}
with open('modelo_con_umbral_PEGADOS.pkl', 'wb') as f:
    pickle.dump(modelo_y_umbral, f)

print("¡Proceso completado con umbral :",str(optimal_threshold))
