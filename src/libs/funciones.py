#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 22:32:29 2023

@author: rockerzega
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Detecta si el equipo posee tarjeta grafica, o solo usa el cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Acciones que realiza la funcion generador
    Genera valores aleatorios para las frecuencias (freq1, freq2) y los desplazamientos (offsets1, offsets2) utilizando np.random.rand().
    Crea un array de tiempo linealmente espaciado de 0 a 1 utilizando np.linspace().
    Genera una serie sinusoidal utilizando las frecuencias y desplazamientos generados. Cada componente sinusoidal se calcula multiplicando la diferencia entre el tiempo y el desplazamiento por la frecuencia correspondiente y aplicando una escala y un desplazamiento adicionales.
    Añade ruido aleatorio a la serie multiplicando un array de ruido aleatorio generado por np.random.rand() por 0.1 y restando 0.5.
    Añade una dimensión adicional a la serie utilizando [..., np.newaxis] para que tenga la forma (batch_size, n_steps, 1).
    Convierte la serie a tipo float32 utilizando .astype(np.float32).
"""


def generador(batch_size, n_steps):
  freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)  # Genera valores aleatorios para las frecuencias y los desplazamientos
  time = np.linspace(0, 1, n_steps)  # Crea un array de tiempo linealmente espaciado de 0 a 1
  series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # Genera una serie sinusoidal con la primera frecuencia y desplazamiento
  series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # Añade otra serie sinusoidal con la segunda frecuencia y desplazamiento
  series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # Añade ruido aleatorio a la serie
  return series[..., np.newaxis].astype(np.float32)  # Añade una dimensión adicional y convierte la serie a tipo float32


"""
Acciones que realiza la funcion graficar
    Define el número de filas (r) y columnas (c) para la disposición de los subplots en una figura.
    Crea una figura (fig) y una matriz de subplots (axes) utilizando plt.subplots().
    Itera sobre las filas (row) y las columnas (col) para crear los subplots y trazar las series de datos correspondientes.
    Selecciona el subplot actual con plt.sca().
    Trazar la serie de datos principal (series[ix, :]) utilizando plt.plot().
    Trazar los puntos reales (y) si están disponibles.
    Trazar las predicciones (y_pred) si están disponibles.
    Trazar la desviación estándar de las predicciones (y_pred_std) si está disponible.
"""


def graficar(series, y=None, y_pred=None, y_pred_std=None, x_label="$t$", y_label="$x$"):
  r, c = 3, 5  # Número de filas y columnas para la disposición de los subplots
  fig, axes = plt.subplots(nrows=r, ncols=c, sharey=True, sharex=True, figsize=(20, 10))  # Creación de la figura y los subplots
  
  # Bucle para crear los subplots y trazar las series de datos
  for row in range(r):
    for col in range(c):
      plt.sca(axes[row][col])  # Selecciona el subplot actual
      ix = col + row*c  # Índice para seleccionar la serie de datos correspondiente
      
      # Trazar la serie de datos
      plt.plot(series[ix, :], ".-")
      
      # Trazar los puntos reales (y) si están disponibles
      if y is not None:
        plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y[ix])), y[ix], "bx", markersize=10)
      
      # Trazar las predicciones (y_pred) si están disponibles
      if y_pred is not None:
        plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix], "ro")
      
      # Trazar la desviación estándar de las predicciones (y_pred_std) si está disponible
      if y_pred_std is not None:
        plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix] + y_pred_std[ix])
        plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix] - y_pred_std[ix])
      
      plt.grid(True)  # Mostrar las líneas de cuadrícula
      plt.hlines(0, 0, 100, linewidth=1)  # Trazar una línea horizontal en y=0
      plt.axis([0, len(series[ix, :])+len(y[ix]), -1, 1])  # Establecer los límites del eje x e y
      
      # Establecer las etiquetas de los ejes x e y
      if x_label and row == r - 1:
        plt.xlabel(x_label, fontsize=16)
      if y_label and col == 0:
        plt.ylabel(y_label, fontsize=16, rotation=0)
  
  plt.show()  # Mostrar la figura con los subplots

"""
Acciones de la funcion entrenar
    Mueve el modelo a la GPU si está disponible.
    Define un optimizador (Adam) y una función de pérdida (MSELoss).
    Itera sobre un rango de épocas.
    Establece el modelo en modo de entrenamiento.
    Itera sobre los lotes de datos de entrenamiento.
    Realiza una predicción con el modelo, calcula la pérdida y realiza la retropropagación y la actualización de los pesos.
    Almacena las pérdidas de entrenamiento en cada época.
    Establece el modelo en modo de evaluación.
    Itera sobre los lotes de datos de evaluación.
    Realiza una predicción con el modelo y calcula la pérdida de evaluación.
    Almacena las pérdidas de evaluación en cada época.
    Actualiza la descripción de la barra de progreso con las pérdidas promedio de entrenamiento y evaluación.
"""
def entrenar(model, dataloader, epochs=10):
  # Mueve el modelo a la GPU si está disponible
  model.to(device)
  
  # Definición del optimizador y función de pérdida
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.MSELoss()
  
  # Creación de una barra de progreso para visualizar el entrenamiento
  bar = tqdm(range(1, epochs+1))
  
  # Bucle principal de entrenamiento por épocas
  for epoch in bar:
    model.train()  # Establece el modelo en modo de entrenamiento
    train_loss = []  # Almacena las pérdidas de entrenamiento
    
    # Bucle de entrenamiento por lotes
    for batch in dataloader['train']:
      X, y = batch
      X, y = X.to(device), y.to(device)
      
      optimizer.zero_grad()  # Reinicia los gradientes acumulados
      y_hat = model(X)  # Realiza una predicción con el modelo
      loss = criterion(y_hat, y)  # Calcula la pérdida
      loss.backward()  # Calcula los gradientes
      optimizer.step()  # Actualiza los pesos del modelo
      
      train_loss.append(loss.item())  # Guarda la pérdida actual
    
    model.eval()  # Establece el modelo en modo de evaluación
    eval_loss = []  # Almacena las pérdidas de evaluación
    
    # Bucle de evaluación por lotes
    with torch.no_grad():
      for batch in dataloader['eval']:
        X, y = batch
        X, y = X.to(device), y.to(device)
        
        y_hat = model(X)  # Realiza una predicción con el modelo
        loss = criterion(y_hat, y)  # Calcula la pérdida
        eval_loss.append(loss.item())  # Guarda la pérdida actual
    
    # Actualiza la descripción de la barra de progreso con las pérdidas promedio
    bar.set_description(f"loss {np.mean(train_loss):.5f} val_loss {np.mean(eval_loss):.5f}")

"""
Acciones de la funcion predecir
    Establece el modelo en modo de evaluación.
    Utiliza el bucle torch.no_grad() para desactivar el cálculo de gradientes y reducir el uso de memoria durante la inferencia.
    Crea un tensor vacío (preds) para almacenar las predicciones.
    Itera sobre los lotes de datos del dataloader.
    Mueve los datos al dispositivo (GPU si está disponible).
    Realiza una predicción con el modelo.
    Concatena las predicciones al tensor preds.
    Devuelve el tensor preds que contiene todas las predicciones.
"""

def predecir(model, dataloader):
  model.eval()  # Establece el modelo en modo de evaluación
  with torch.no_grad():
    preds = torch.tensor([]).to(device)  # Tensor vacío para almacenar las predicciones
    
    # Bucle de predicción por lotes
    for batch in dataloader:
      X = batch
      X = X.to(device)
      
      pred = model(X)  # Realiza una predicción con el modelo
      preds = torch.cat([preds, pred])  # Concatena las predicciones al tensor de predicciones
    
    return preds  # Devuelve el tensor de predicciones

"""
Funciun entrenarc, entrenamiento para modelo colapsable

Descripcion

    La función fit toma como entrada el modelo (model) que se desea entrenar, 
    el dataloader (dataloader) que contiene los datos de entrenamiento y evaluación, 
    y el número de épocas (epochs) que se utilizarán para entrenar el modelo. 
    El modelo se mueve al dispositivo especificado por device.

    Se crea un optimizador Adam (optimizer) que se encargará de actualizar los 
    parámetros del modelo durante el entrenamiento. Se utiliza una función de 
    pérdida de error cuadrático medio (criterion) para calcular la pérdida 
    durante el entrenamiento.

    Se itera sobre cada época (for epoch in bar) y se inicializa una lista 
    vacía para almacenar la pérdida de entrenamiento (train_loss) y otra lista 
    para almacenar la pérdida del último paso (train_loss2). Se configura el 
    modelo en modo de entrenamiento (model.train()).

    Dentro del bucle de entrenamiento, se itera sobre los lotes del dataloader 
    de entrenamiento (for batch in dataloader['train']) y se obtienen las 
    características de entrada (X) y las etiquetas de salida (y) del lote. 
    Se mueven al dispositivo especificado y se restablecen los gradientes del 
    optimizador (optimizer.zero_grad()).

    Se realiza una predicción utilizando el modelo (y_hat = model(X)) y se 
    calcula la pérdida utilizando la función de pérdida definida 
    (loss = criterion(y_hat, y)). Luego, se realiza la retropropagación del 
    error (loss.backward()) y se actualizan los parámetros del modelo mediante 
    el optimizador (optimizer.step()). Se registra la pérdida de entrenamiento 
    y la pérdida del último paso en las listas correspondientes.
    
    Finalmente, se actualiza la barra de progreso con información sobre las 
    pérdidas de entrenamiento y evaluación. La información incluye la pérdida 
    promedio de entrenamiento (np.mean(train_loss)), la pérdida del último paso 
    promedio de entrenamiento (np.mean(train_loss2)), la pérdida promedio de 
    evaluación (np.mean(eval_loss)) y la pérdida del último paso promedio de 
    evaluación (np.mean(eval_loss2))
"""

def entrenarc(model, dataloader, epochs=10):
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.MSELoss()
  bar = tqdm(range(1, epochs+1))
  for epoch in bar:
    model.train()
    train_loss = []
    train_loss2 = []
    for batch in dataloader['train']:
      X, y = batch
      X, y = X.to(device), y.to(device)
      optimizer.zero_grad()
      y_hat = model(X)
      loss = criterion(y_hat, y)
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())
      train_loss2.append((y[:,-1] - y_hat[:,-1]).pow(2).mean().item())
    model.eval()
    eval_loss = []
    eval_loss2 = []
    with torch.no_grad():
      for batch in dataloader['eval']:
        X, y = batch
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = criterion(y_hat, y)
        eval_loss.append(loss.item())
        eval_loss2.append((y[:,-1] - y_hat[:,-1]).pow(2).mean().item())
    bar.set_description(f"loss {np.mean(train_loss):.5f} loss_last_step {np.mean(train_loss2):.5f} val_loss {np.mean(eval_loss):.5f} val_loss_last_step {np.mean(eval_loss2):.5f}")

"""
Como la regularizacion con el dropout apagara las neuronas de manera aleatoria
para el entrenamiento necesitamos que toda las neuronas esten encendidas

creamos la funcion predictIC
"""

def predecirIC(model, dataloader):
    # activar dropout para evaluación !
    model.train() # lo normal aquí es poner model.eval()
    with torch.no_grad():
        preds = torch.tensor([]).to(device)
        for batch in dataloader:
            X = batch
            X = X.to(device)
            pred = model(X)
            preds = torch.cat([preds, pred])
        return preds