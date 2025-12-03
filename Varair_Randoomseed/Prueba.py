import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. Simulemos una curva de resistencia TÍPICA con un poco de ruido eléctrico
t = np.linspace(0, 100, 100)
# Curva base: sube, pico, baja
y_real = -0.01*(t-40)**2 + 200 
# Añadimos ruido aleatorio (típico de sensores)
ruido = np.random.normal(0, 2, 100) 
y_con_ruido = y_real + ruido

# 2. Método SIN filtro (Tu código original)
derivada_cruda = np.gradient(np.gradient(y_con_ruido)) # 2da derivada

# 3. Método CON filtro (Mi propuesta)
y_suave = savgol_filter(y_con_ruido, window_length=15, polyorder=3)
derivada_filtrada = np.gradient(np.gradient(y_suave)) # 2da derivada

# 4. Graficar
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, y_con_ruido, label='Señal Sensor (Ruidosa)', alpha=0.5)
plt.plot(t, y_suave, label='Señal Filtrada (Savitzky-Golay)', color='red')
plt.title("1. La señal de Resistencia")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, derivada_cruda, label='2ª Derivada ORIGINAL (Sin filtro)', color='grey')
plt.plot(t, derivada_filtrada, label='2ª Derivada PROPUESTA (Con filtro)', color='red', linewidth=2)
plt.title("2. Lo que ve tu modelo (Segunda Derivada)")
plt.legend()
plt.tight_layout()
plt.show()