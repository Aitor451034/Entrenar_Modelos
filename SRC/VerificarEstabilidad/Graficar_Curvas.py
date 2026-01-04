import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tkinter as tk
from tkinter import filedialog
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. FUNCIONES DE CARGA Y PROCESAMIENTO
# ==============================================================================

def leer_archivo():
    print("Selecciona el archivo CSV de datos...")
    root = tk.Tk(); root.withdraw()
    ruta = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
    if not ruta: return None
    try:
        return pd.read_csv(ruta, encoding="utf-8", sep=";", on_bad_lines="skip", 
                           header=None, dtype=str, skiprows=3)
    except Exception as e:
        print(f"Error: {e}")
        return None

def preprocesar_dataframe(df_raw):
    if df_raw.shape[1] < 99: return None
    new_df = df_raw.iloc[:, [0, 8, 9, 10, 20, 27, 67, 98]].copy()
    new_df = new_df.iloc[:-2]
    new_df.columns = ["id", "Ns", "Corrientes", "Voltajes", "KAI2", "Ts2", "Fuerza", "Etiqueta"]
    
    for col in ["Ns", "Ts2", "Etiqueta"]:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int)

    new_df = new_df.drop_duplicates()
    new_df.index = range(1, len(new_df) + 1)
    return new_df

def obtener_curva_procesada(row):
    try:
        str_volt = row["Voltajes"]; str_corr = row["Corrientes"]
        if pd.isna(str_volt) or pd.isna(str_corr): return None, None

        raw_volt = np.array([float(v.replace(',', '.')) for v in str_volt.split(';') if v.strip()])
        raw_corr = np.array([float(v.replace(',', '.')) for v in str_corr.split(';') if v.strip()])
        
        ns = int(row["Ns"]); ts2 = float(row["Ts2"])
        t_soldadura = np.linspace(0, ts2, ns + 1)
        
        min_len = min(len(t_soldadura), len(raw_volt), len(raw_corr))
        if min_len < 10: return None, None
        
        t_soldadura = t_soldadura[:min_len]
        raw_volt = raw_volt[:min_len]
        raw_corr = raw_corr[:min_len]

        # Recorte simple
        start_idx = 0
        while start_idx < len(raw_corr) and raw_corr[start_idx] < 150.0: start_idx += 1
        
        if len(raw_volt) - start_idx < 10: return None, None

        raw_volt = raw_volt[start_idx:]
        raw_corr = raw_corr[start_idx:]
        t_soldadura = t_soldadura[start_idx:]
        t_soldadura = t_soldadura - t_soldadura[0]

        epsilon = 1e-5
        vals_R = np.divide(raw_volt, raw_corr + epsilon, out=np.zeros_like(raw_volt), where=np.abs(raw_corr)>epsilon)
        vals_R[np.isinf(vals_R)] = 0.0
        
        # Suavizado suave
        window = 7 if len(vals_R) > 7 else 3
        r_smooth = savgol_filter(vals_R, window_length=window, polyorder=2)
            
        return t_soldadura, r_smooth
    except:
        return None, None

def normalizar_z_score(signal):
    std = np.std(signal)
    if std < 1e-6: return np.zeros_like(signal) # Señal plana = 0
    return (signal - np.mean(signal)) / std

def es_curva_coherente(r_signal):
    """
    FILTRO DE CALIDAD: Devuelve True solo si la curva parece una soldadura real.
    Descarta líneas planas, ruido puro o ceros.
    """
    if r_signal is None or len(r_signal) < 5: return False
    
    r_max = np.max(r_signal)
    r_mean = np.mean(r_signal)
    r_std = np.std(r_signal)
    
    # Criterio 1: No debe ser plana (Desviación estándar mínima)
    if r_std < 0.01: return False 
    
    # Criterio 2: Debe tener una resistencia promedio razonable (evitar cables sueltos)
    # Ajusta este valor si tus resistencias son muy bajas, pero < 0.1 solía ser ruido en tus gráficos
    if r_mean < 0.1: return False
    
    return True

# ==============================================================================
# 2. VISUALIZACIÓN
# ==============================================================================

def main():
    df_raw = leer_archivo()
    if df_raw is None: return
    df = preprocesar_dataframe(df_raw)
    
    print(f"\nDatos cargados. Total: {len(df)}")
    
    while True:
        try:
            val = input("\nID a analizar (ej. 160) [q para salir]: ").strip()
            if val.lower() == 'q': break
            target_idx = int(val)
        except: continue

        if target_idx not in df.index: print("ID no existe"); continue

        # 1. Target
        row_target = df.loc[target_idx]
        t_t, r_t = obtener_curva_procesada(row_target)
        if t_t is None: print("Curva Target inválida"); continue
        
        lbl_target = "DEFECTO (Pegado)" if row_target["Etiqueta"]==1 else "OK"
        col_target = 'red' if row_target["Etiqueta"]==1 else 'blue'

        # 2. Referencias INTELIGENTES
        # Buscamos soldaduras OK que pasen el filtro de calidad
        refs = []
        candidatos = df[df["Etiqueta"]==0].sample(frac=1).index.tolist() # Barajar
        
        for idx in candidatos:
            if idx == target_idx: continue
            t_ref, r_ref = obtener_curva_procesada(df.loc[idx])
            
            # ¡AQUÍ ESTÁ LA MAGIA! Solo aceptamos curvas coherentes
            if es_curva_coherente(r_ref):
                refs.append((t_ref, r_ref))
            
            if len(refs) >= 5: break # Solo necesitamos 5 buenas para comparar

        # 3. Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Izquierda: Absoluto
        for tr, rr in refs: ax1.plot(tr, rr, color='gray', alpha=0.3)
        ax1.plot(t_t, r_t, color=col_target, linewidth=2.5, label=f"Target {target_idx}")
        ax1.set_title(f"Valores Reales (Ref: {len(refs)} curvas OK limpias)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Derecha: Normalizado
        for tr, rr in refs: ax2.plot(tr, normalizar_z_score(rr), color='gray', alpha=0.3)
        ax2.plot(t_t, normalizar_z_score(r_t), color=col_target, linewidth=2.5)
        ax2.set_title("Forma Normalizada (Comparación de Dinámica)")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Análisis: ID {target_idx} ({lbl_target})", fontsize=14)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()