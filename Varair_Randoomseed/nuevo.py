import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis

# ==========================================
# 1. TU FUNCIÓN DE CARGA
# ==========================================
def leer_archivo():
    """Abre un diálogo para seleccionar un archivo CSV y lo carga como DataFrame."""
    print("Selecciona el archivo CSV que contiene los datos...")
    root = tk.Tk()
    root.withdraw() # Ocultar la ventana principal de TK

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
        
        # Preprocesamiento básico de columnas según tu estructura
        # Asumiendo que las columnas relevantes están en estas posiciones fijas
        # Ajusta estos índices si tu CSV cambia
        new_df = df.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]]
        new_df = new_df.iloc[:-2] # Quitar filas finales basura
        new_df.columns = ["id punto", "Ns", "Corrientes inst.", "Voltajes inst.", "KAI2", "Ts2", "Fuerza", "Etiqueta datos"]
        new_df.index = range(len(new_df))
        
        print("¡Archivo CSV leído y estructurado correctamente!")
        return new_df
    except Exception as e:
        print(f"Error al leer o procesar el archivo: {e}")
        return None

# ==========================================
# 2. FUNCIÓN DE AUDITORÍA COMPARATIVA
# ==========================================
def auditoria_comparativa(df, indice_fila=0):
    """
    Toma una fila del DF y calcula los parámetros con el método ANTIGUO
    y el método NUEVO para ver las diferencias.
    """
    
    # --- A. PREPARACIÓN DE DATOS (COMÚN) ---
    if indice_fila >= len(df):
        print(f"El índice {indice_fila} no existe en el DataFrame.")
        return

    row = df.iloc[indice_fila]
    
    str_volt = row["Voltajes inst."]
    str_corr = row["Corrientes inst."]
    
    if pd.isna(str_volt) or pd.isna(str_corr):
        print("La fila seleccionada tiene datos nulos.")
        return

    # Parseo básico
    try:
        raw_volt = np.array([float(v) for v in str(str_volt).split(';') if v.strip()])
        raw_corr = np.array([float(v) for v in str(str_corr).split(';') if v.strip()])
    except ValueError:
        print("Error parseando los strings de voltaje/corriente.")
        return
    
    try:
        ns = int(row["Ns"])
        ts2 = int(row["Ts2"])
    except:
        ns = len(raw_volt)
        ts2 = len(raw_volt) 
        
    t_soldadura = np.linspace(0, ts2, ns + 1)
    
    # Recorte para igualar longitudes
    min_len = min(len(t_soldadura), len(raw_volt), len(raw_corr))
    if min_len == 0:
        print("Datos vacíos tras el recorte.")
        return

    t = t_soldadura[:min_len]
    v = raw_volt[:min_len]
    i_raw = raw_corr[:min_len] # kA
    
    print(f"\n--- ANALIZANDO PUNTO ID (Fila {indice_fila}) ---")
    
    # ==========================================================================
    # MÉTODO 1: TU CÓDIGO (Lógica Original replicada)
    # ==========================================================================
    # Tu lógica de limpieza de corriente
    i_old = np.array([0.001 if x == 0 else x for x in i_raw])
    v_old = np.array([round(x, 0) for x in v]) # Tu redondeo
    i_old = np.array([round(x, 0) for x in i_old])
    
    # Tu cálculo de resistencia
    r_old = []
    for volts, amps in zip(v_old, i_old):
        r_old.append(volts / amps if amps != 0 else 0)
    r_old = np.array([round(x, 2) for x in r_old])
    
    # Tus Derivadas (Crudas)
    d1_old = np.gradient(r_old, t)
    d2_old = np.gradient(d1_old, t)
    d3_old = np.gradient(d2_old, t)
    
    # Tus Features Clave
    max_curv_old = np.max(np.abs(d2_old))
    max_jerk_old = np.max(np.abs(d3_old))
    
    # Tu Energía (Unidad incorrecta kA*10)
    i_amp_old = i_old * 10 
    t_sec_old = t / 1000.0
    pot_old = (v_old/100.0) * i_amp_old
    Q_old = np.trapz(pot_old, x=t_sec_old)
    
    # Tu Pendiente OLS (Fórmula incorrecta con varianza de R)
    t_mean = np.mean(t)
    r_mean = np.mean(r_old)
    # Replicando tu lógica exacta con bucle sum:
    num = sum((r_mean - ri) * (t_mean - ti) for ri, ti in zip(r_old, t))
    den = sum((r_mean - ri) ** 2 for ri in r_old) # EL ERROR
    m_ols_old = num / den if den != 0 else 0

    # ==========================================================================
    # MÉTODO 2: MI CÓDIGO (Corregido y Optimizado)
    # ==========================================================================
    # Filtrado inicial 
    r_new_raw = np.divide(v, i_raw, out=np.zeros_like(v), where=np.abs(i_raw)>0.5)
    
    # Filtro Savitzky-Golay
    window = min(11, len(r_new_raw) if len(r_new_raw)%2!=0 else len(r_new_raw)-1)
    if window > 3:
        r_new = savgol_filter(r_new_raw, window_length=window, polyorder=3)
    else:
        r_new = r_new_raw

    # Derivadas (Suavizadas)
    d1_new = np.gradient(r_new, t)
    d2_new = np.gradient(d1_new, t)
    d3_new = np.gradient(d2_new, t)
    
    max_curv_new = np.max(np.abs(d2_new))
    max_jerk_new = np.max(np.abs(d3_new))
    
    # Energía Corregida (kA * 1000)
    i_amp_new = i_raw * 1000.0
    t_sec_new = t / 1000.0
    pot_new = (v/100.0) * i_amp_new
    Q_new = np.trapz(pot_new, x=t_sec_new)
    
    # Pendiente OLS Corregida (Varianza de T)
    t_mean = np.mean(t)
    r_mean_new = np.mean(r_new)
    num_new = np.sum((r_new - r_mean_new) * (t - t_mean))
    den_new = np.sum((t - t_mean) ** 2) # CORREGIDO
    m_ols_new = num_new / den_new if den_new != 0 else 0

    # ==========================================================================
    # 3. VISUALIZACIÓN DE RESULTADOS
    # ==========================================================================
    
    # Tabla Comparativa (Usando print normal para evitar error de tabulate)
    print("\n" + "="*80)
    print(f"{'PARÁMETRO':<25} | {'TU CÓDIGO (ORIGINAL)':<20} | {'MI CÓDIGO (NUEVO)':<20} | {'NOTA'}")
    print("-" * 80)
    print(f"{'Energía (J)':<25} | {Q_old:<20.2f} | {Q_new:<20.2f} | {'x100 (Unidades)'}")
    print(f"{'Pendiente Tendencia':<25} | {m_ols_old:<20.6f} | {m_ols_new:<20.6f} | {'Fórmula corregida'}")
    print(f"{'Max Curvatura':<25} | {max_curv_old:<20.4f} | {max_curv_new:<20.4f} | {'Filtrado vs Ruido'}")
    print(f"{'Max Jerk':<25} | {max_jerk_old:<20.4f} | {max_jerk_new:<20.4f} | {'Filtrado vs Ruido'}")
    print("="*80 + "\n")
    
    # Gráficos
    plt.figure(figsize=(14, 10))
    
    # Gráfico 1: Resistencia
    plt.subplot(2, 2, 1)
    plt.plot(t, r_old, label='Tu Código (Sin filtro)', alpha=0.5, color='gray')
    plt.plot(t, r_new, label='Mi Código (Savitzky-Golay)', color='red', linewidth=2)
    plt.title("1. Calidad de Señal (Resistencia)")
    plt.ylabel("Resistencia")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: 2ª Derivada (Curvatura)
    plt.subplot(2, 2, 2)
    plt.plot(t, d2_old, label='Tuya (Ruido puro)', alpha=0.5, color='gray')
    plt.plot(t, d2_new, label='Mía (Detecta física)', color='blue', linewidth=2)
    plt.title("2. Segunda Derivada (Curvatura)")
    plt.legend()
    
    # Gráfico 3: Pendiente OLS
    plt.subplot(2, 2, 3)
    plt.plot(t, r_new, color='red', alpha=0.3)
    # Rectas
    y_old_line = m_ols_old * (t - t_mean) + r_mean
    y_new_line = m_ols_new * (t - t_mean) + r_mean_new
    
    plt.plot(t, y_old_line, label=f'Tu Recta (m={m_ols_old:.4f})', linestyle='--', color='gray')
    plt.plot(t, y_new_line, label=f'Mi Recta (m={m_ols_new:.4f})', linestyle='--', color='green')
    plt.title("3. Tendencia Lineal (Mínimos Cuadrados)")
    plt.legend()
    
    # Gráfico 4: Energía
    plt.subplot(2, 2, 4)
    plt.bar(['Tu Energía', 'Energía Real'], [Q_old, Q_new], color=['gray', 'orange'])
    plt.title("4. Diferencia de Escala en Energía")
    plt.ylabel("Joules")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Cargar datos
    mi_df = leer_archivo()
    
    # 2. Si se cargó bien, ejecutar auditoría
    if mi_df is not None:
        print(f"DataFrame cargado con {len(mi_df)} filas.")
        
        # Puedes cambiar el índice (0, 1, 2...) para ver diferentes soldaduras
        indice_a_probar = 0 
        
        auditoria_comparativa(mi_df, indice_fila=indice_a_probar)