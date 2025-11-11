
# Introduction:
Esta carpeta contiene la información del modelo desarrollado para la identificación de faltas de fusiones de puntos de soldadura
mediante el análisis de las curvas de resistencia dinámica de los puntos. Para ello se analiza el comportamiento de 32 parámetros
estadísticos de las curvas.

# Archivos:
Breve explicación de para que sirve cada código:
1.	entrenar_modelo_pegados_titanio: en este código se extrae los datos de un archivo en formato .CSV, se realizan las transformaciones
									 de los datos (cálculo de los 32 parámetros estadísticos), se crea el modelo, se entrena y se 
									 calcula la matriz de confusión, la curva roc y el classification report evaluados en el test set.
2.	modelo_PEGADOS.pkl: archivo en el que se guarda el modelo creado en el código anterior.
3. 	modelo_con_umbral_PEGADOS.pkl: archivo en el que se almacena tanto el modelo como el mejor umbral de predicciones optimizado según el mejor		
						   f2-score evaluado con el train set.
4.	RESULTADOS_MODELO_PEGADOS.md: recoge todos los resultados obtenidos con este modelo.
5.	Inputs_modelo_pegado.csv: archivo csv que contiene los inputs del modelo (Rawdata).



