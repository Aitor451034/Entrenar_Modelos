import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc, fbeta_score, make_scorer, classification_report, confusion_matrix
import seaborn as sns
from sklearn import metrics
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import pickle
#--------------------------------------------AÑADIDO ADICIONAL
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#------------------------------------------------
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
        ruta_csv = r"C:\Users\U5014554\Desktop\EntrenarModelo\DATA\Inputs_modelo_pegado_con_datos2.csv"
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
X = X.map(lambda x: round(x, 4))
print(X)
y = pd.Series(y)
# ==============================================================================


# Separar en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42, stratify=y)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Crear StratifiedKFold para validación cruzada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definir el scorer para F2-score
fbeta = 5
f2_scorer = make_scorer(fbeta_score, beta=fbeta)

# Lista de valores para C (inverso de lambda)
lista_lambdas = np.linspace(0.0001, 10, 10000)

# Crear pipeline con escalado y regresión logística
modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l1", tol=1e-5, solver="liblinear", max_iter=50000, random_state=42)
)

# Importante: en param_grid usar el nombre del paso + '__' + nombre parámetro
param_grid = {"logisticregression__C": lista_lambdas}

# Configurar GridSearchCV con stratified K-fold y F2 scorer
grid_search = GridSearchCV(
    estimator=modelo,
    param_grid=param_grid,
    cv=skf,
    scoring=f2_scorer,
    n_jobs=-1,
    verbose=1
)

# Ejecutar búsqueda
grid_search.fit(X_train, y_train)

# Obtener mejores parámetros y modelo
mejor_lambda = grid_search.best_params_["logisticregression__C"]
mejor_modelo = grid_search.best_estimator_

# Ajustar el mejor modelo al conjunto completo de entrenamiento (opcional, ya lo hizo GridSearch)
mejor_modelo.fit(X_train, y_train)

# Guardar modelo
joblib.dump(mejor_modelo, "modelo_PEGADOS.pkl")

print(f"Mejor lambda (C): {mejor_lambda}")



# ==============================================================================
# Resultados del modelo
mejor_lambda = grid_search.best_params_["logisticregression__C"]
print(f"Mejor lambda: {mejor_lambda}")
mejor_score = grid_search.best_score_
print(f"Mejor score: {mejor_score}")
# ==============================================================================
# Coeficientes del modelo
df_coeficientes1 = pd.DataFrame({'predictor': X_train.columns,'coef': mejor_modelo.named_steps['logisticregression'].coef_.flatten()})
# Predictores incluidos en el modelo (coeficiente != 0)
df_coeficientes1[df_coeficientes1.coef != 0]
print(df_coeficientes1.coef)
df_coeficientes_no_cero1 = df_coeficientes1[np.abs(df_coeficientes1.coef) > 0]
n_predictores1 = len(df_coeficientes_no_cero1)
fig, ax = plt.subplots(figsize=(11, 3.84))
ax.stem(df_coeficientes1.predictor, df_coeficientes1.coef, markerfmt=' ')
plt.xticks(rotation=90, ha='right', size=5)
ax.set_xlabel('variable')
ax.set_ylabel('coeficientes')
ax.set_title(f'Coeficientes del modelo con {n_predictores1} predictores')
for i, row in df_coeficientes_no_cero1.iterrows():
    predictor_label = int(row['predictor'])
    ax.annotate(f"({predictor_label}, {row['coef']:.6f})",
                xy=(row['predictor'], row['coef']),
                xytext=(0, 5 if row['coef'] > 0 else -5),  
                textcoords="offset points",
                ha='center', va='bottom' if row['coef'] > 0 else 'top',
                arrowprops=dict(arrowstyle='-', color='black'))
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
plt.title("Confusion matrix MODELO LASSO USANDO REGRESION LOGISTICA sin optimizar el umbral de predicciones (umbral = 0.5)")
plt.ylabel('Etiqueta real')
plt.xlabel('Predicción')
plt.show()
# ==============================================================================
# Calcular la matriz de confusión optimizando el umbral de predicciones
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
lista_umbrales = np.linspace(0, 1, 1000)

# Acumular probabilidades y verdaderas etiquetas de todas las validaciones
# Paso 1: Entrenar el modelo con todo X_train y y_train
mejor_modelo.fit(X_train, y_train)

# Paso 2: Obtener probabilidades en conjunto de validación (ejemplo: X_test)
probas_validacion = mejor_modelo.predict_proba(X_test)[:, 1]

# Paso 3: Buscar umbral óptimo sobre validación
lista_umbrales = np.linspace(0, 0.5, 4000)
best_fbeta_score = -1
optimal_threshold = 0
fbeta = 2

for thresh in lista_umbrales:
    preds_bin = np.where(probas_validacion >= thresh, 1, 0)
    score = fbeta_score(y_test, preds_bin, beta=fbeta)
    if score > best_fbeta_score:
        best_fbeta_score = score
        optimal_threshold = thresh

print(f"Umbral óptimo encontrado: {optimal_threshold:.4f} con F2-score: {best_fbeta_score:.4f}")

# Aplicar umbral en test
predicciones_test_proba = mejor_modelo.predict_proba(X_test)[:, 1]
predicciones_test_binarias = np.where(predicciones_test_proba >= optimal_threshold, 1, 0)
# Resetear el indice de y_test_new para poder indexarlo correctamente
y_test_new_reset = y_test.reset_index(drop=True)
# Mostrar las predicciones de probabilidades de pertenencia y las etiquetas reales
for i in range(len(predicciones_test_proba)):
    print(f"Predicción de probabilidad para el punto {i}: {predicciones_test_proba[i]:.4f}, real: {y_test_new_reset[i]}")
# ==============================================================================
# Calcular la matriz de confusión con el umbral óptimo
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
plt.title(f"Confusion matrix MODELO LASSO USANDO REGRESION LOGISTICA optimizando el umbral de predicciones = {optimal_threshold:.4f}")
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
# Guardar modelo y umbral de predicciones en un mismo .pkl
print(optimal_threshold)
modelo_y_umbral = { "modelo": mejor_modelo, "umbral de predicciones": optimal_threshold}
with open('modelo_con_umbral_PEGADOS.pkl', 'wb') as f:
    pickle.dump(modelo_y_umbral, f)
