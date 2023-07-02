#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICCION DE UNA SERIE TEMPORAL

"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from libs.funciones import generador, graficar, entrenar, predecir, entrenarc, predecirIC
from libs.clases import DataSerieTemporal, RNN, DeepRNN, DeepRNNM, DeepRNNC, DeepRNNIC

n_steps = 50
series = generador(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

# Mostramos la informacion de las series
print(X_train.shape, y_train.shape)

graficar(X_test, y_test)

dataset = {
  'train': DataSerieTemporal(X_train, y_train),
  'eval': DataSerieTemporal(X_valid, y_valid),
  'test': DataSerieTemporal(X_test, y_test, train=False)
}

dataloader = {
  'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
  'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
  'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}

rnn = RNN()

entrenar(rnn, dataloader)

y_pred = predecir(rnn, dataloader['test'])
graficar(X_test, y_test, y_pred.cpu().numpy())
# Muestra el error cuadratico medio
print(mean_squared_error(y_test, y_pred.cpu()))

drnn = DeepRNN()
entrenar(drnn, dataloader)

y_pred = predecir(drnn, dataloader['test'])
graficar(X_test, y_test, y_pred.cpu().numpy())
print(mean_squared_error(y_test, y_pred.cpu()))

"""
Prediccion con 10 predicciones siguientes, util para determinar probables
valores futuros
"""

# Número de pasos de tiempo en la secuencia generada
n_steps = 50
n_future = 10

# Genera una serie de tiempo con 10000 muestras y `n_steps + n_future` pasos de tiempo, predicciones a futuro
series = generador(10000, n_steps + n_future)

# Conjunto de entrenamiento: toma las primeras 7000 muestras de la serie como entrada y las últimas `n_future` muestras como objetivo
X_train, Y_train = series[:7000, :n_steps], series[:7000, -n_future:, 0]

# Conjunto de validación: toma las muestras de 7000 a 9000 de la serie
# como entrada y las últimas 1`n_future` muestras como objetivo
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -n_future:, 0]

# Conjunto de prueba: toma las muestras a partir de la posición 9000 de la serie 
# como entrada y las últimas `n_future` muestras como objetivo
X_test, Y_test = series[9000:, :n_steps], series[9000:, -n_future:, 0]

dataset = {
    'train': DataSerieTemporal(X_train, Y_train),
    'eval': DataSerieTemporal(X_valid, Y_valid),
    'test': DataSerieTemporal(X_test, Y_test, train=False)
}

dataloader = {
    'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
    'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
    'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}

# Graficamos los resultados con `n_future` predicciones
graficar(X_test, Y_test)

# Concatenamos la serie con los `n_future` valores reales con la prediccion anterior
X = X_test
for step_ahead in range(10):
  inputs = torch.from_numpy(X[:, step_ahead:]).unsqueeze(0)
  y_pred_one = predecir(rnn, inputs).cpu().numpy()
  X = np.concatenate([X, y_pred_one[:, np.newaxis, :]], axis=1)

y_pred = X[:, n_steps:, -1]
graficar(X_test, Y_test, y_pred)

# Mostrar el Error Cuadratico Medio
print(mean_squared_error(Y_test, y_pred))

"""
Mejorando el modelo 
"""

rnn = DeepRNNM()
entrenar(rnn, dataloader)

y_pred = predecir(rnn, dataloader['test'])
graficar(X_test, Y_test, y_pred.cpu().numpy())

# Muestra el error cuadratico medio, el cual es menor al anterior
print(mean_squared_error(Y_test, y_pred.cpu()))

"""
Se observo que el modelo mejoro si se calcula la salida en eñ instante tn-1
asi que, podemos mejorar el modelo lo optimizamos en cada instante t

Colocaremos otra nueva data para proceder el mejoramiento
"""

n_steps = 50
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((10000, n_steps, 10), dtype=np.float32)
for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

dataset = {
    'train': DataSerieTemporal(X_train, Y_train),
    'eval': DataSerieTemporal(X_valid, Y_valid),
    'test': DataSerieTemporal(X_test, Y_test, train=False)
}

dataloader = {
    'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
    'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
    'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}

rnn = DeepRNNC()
entrenarc(rnn, dataloader)

y_pred = predecir(rnn, dataloader['test'])
graficar(X_test, Y_test[:,-1], y_pred[:,-1].cpu().numpy())

# Imprimimos el error cuadratico medio para ver si el modelo ha mejorado
print(mean_squared_error(Y_test[:,-1], y_pred[:,-1].cpu()))

"""
Modelo con intervalos de confianza

Las precciones al ser probabilidades, es usual establecer intervalos de codnfianza
el metodo a usar es el mas simple el dropout, Torch ya tiene integrada la capa
dropout, asi que directamente la usaremos
"""

rnn = DeepRNNIC(dropout=0.3)
entrenarc(rnn, dataloader)


y_preds = np.stack([predecirIC(rnn, dataloader['test']).cpu().numpy() for sample in range(100)])
y_pred = y_preds.mean(axis=0)
y_pred_std = y_preds.std(axis=0)


graficar(X_test, Y_test[:,-1], y_pred[:,-1], y_pred_std[:, -1])