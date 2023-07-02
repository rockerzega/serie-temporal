#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 23:15:44 2023

@author: rockerzega
"""

import torch
from torch.utils.data import Dataset

class DataSerieTemporal(Dataset):
  def __init__(self, X, y=None, train=True):
    self.X = X
    self.y = y
    self.train = train

  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):
    if self.train:
      return torch.from_numpy(self.X[ix]), torch.from_numpy(self.y[ix])
    return torch.from_numpy(self.X[ix])

class RNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
    self.fc = torch.nn.Linear(20, 1)

  def forward(self, x):
    x, h = self.rnn(x) 
    y = self.fc(x[:,-1])
    return y

"""
Descripcion de la clase DeepRNN
    En el método __init__(), se inicializa la clase padre torch.nn.Module y se define la arquitectura de la red.
    Se define una capa RNN utilizando torch.nn.RNN. La capa tiene un tamaño de entrada de 1 (cada paso de tiempo en la secuencia se considera como un solo valor), un tamaño oculto de 20, 2 capas ocultas y se configura para operar en el modo de lote primero (batch_first=True).
    Se define una capa lineal (torch.nn.Linear) que mapea el último estado oculto de la capa RNN a una salida de 1 dimensión.
    En el método forward(), se realiza la propagación hacia adelante de la red.
    La secuencia de entrada x se pasa a través de la capa RNN. Se devuelve tanto la secuencia de salida como el estado oculto final (h), aunque en este caso solo se utiliza la secuencia de salida.
    La secuencia de salida de la capa RNN se pasa a través de la capa lineal para obtener la salida final. Se selecciona el último estado oculto (x[:,-1]) ya que es el que contiene la información más reciente de la secuencia.
    La salida final se devuelve como resultado de la función forward().
"""

class DeepRNN(torch.nn.Module):
  def __init__(self, n_in=50, n_out=1, num_layers=2):
    super().__init__()
    # Capa RNN con `num_layers` capas ocultas y un tamaño de entrada de 1
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=num_layers, batch_first=True)
    # Capa lineal que mapea el último estado oculto a una salida de 1 dimensión
    self.fc = torch.nn.Linear(20, 1)

  def forward(self, x):
    # Capa lineal que mapea el último estado oculto a una salida de 1 dimensión
    x, h = self.rnn(x)
    # Pasa el último estado oculto a través de la capa lineal para obtener la salida final
    x = self.fc(x[:,-1])
    return x

"""
Descripcion del DeepRNNM

    n_out: indica el número de salidas de la red. 
    Se crea una capa RNN con una entrada de tamaño 1, un tamaño oculto de 20 y 2 capas ocultas.
    También se crea una capa lineal (fully connected) con una entrada de tamaño 20 
    y una salida de tamaño n_out.
    
    El método forward implementa el flujo hacia adelante (forward pass) de la red. 
    
    Recibe un tensor x como entrada, que representa la secuencia de entrada de 
    la serie de tiempo. Dentro del método, se aplica la capa RNN self.rnn a x, 
    lo que produce dos salidas: x y h. Aquí, 
    x contiene las representaciones de salida de cada paso de tiempo, 
    y h es el estado oculto final de la RNN. 
    
    Luego, se toma la última salida de x utilizando el índice x[:, -1]. 
    
    Finalmente, se aplica la capa lineal self.fc a la última salida de la RNN 
    para obtener la salida final de la red.
"""

class DeepRNNM(torch.nn.Module):
  def __init__(self, n_out=10):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=2, batch_first=True)
    self.fc = torch.nn.Linear(20, n_out)

  def forward(self, x):
    x, h = self.rnn(x) 
    x = self.fc(x[:,-1])
    return x

"""
RNN que colapsa su dimension

Conecta las neuronas de la capa oculta en el último paso con las salidas. 
Para poder optimizar para todos los instantes tenemos que colapsar las dimensiones 
del batch y los instantes temporales de la manera que puedes ver a continuación.
"""

class DeepRNNC(torch.nn.Module):
  def __init__(self, n_out=10):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=2, batch_first=True)
    self.fc = torch.nn.Linear(20, n_out)

  def forward(self, x):
    x, h = self.rnn(x) 
    x_reshaped = x.contiguous().view(-1, x.size(-1))
    y = self.fc(x_reshaped)
    y = y.contiguous().view(x.size(0), -1, y.size(-1))
    return y

"""
RNN que colapsa y tiene intervalo de confianza

Descripcion de la clas DeepRNNIC

    El método __init__ se ejecuta al crear una instancia de la clase DeepRNN. 
    Toma como parámetros n_out y dropout, donde n_out especifica la dimensión 
    de salida y dropout indica la probabilidad de dropout para las capas RNN. 
    Llama al constructor de la clase base (super().__init__()) para inicializar 
    la clase base torch.nn.Module.

    Dentro del método __init__, se define un objeto rnn que es una capa RNN. 
    Utiliza el tipo de celda RNN por defecto con una entrada de tamaño 1, un 
    tamaño oculto de 20, 2 capas y batch_first=True, lo que indica que los datos 
    tienen la forma de (batch_size, sequence_length, input_size). Se puede 
    especificar un valor de dropout para regularizar la capa RNN.

    También se define un objeto fc que es una capa lineal (fully connected) que 
    toma como entrada el tamaño oculto de la capa RNN (20) y produce una salida 
    de tamaño n_out.

    El método forward define el flujo de datos hacia adelante en el modelo. 
    Toma una entrada x y la pasa a través de la capa RNN (self.rnn(x)), lo que 
    produce una salida x y un estado oculto h. Luego, se remodela la salida x a 
    una forma conveniente para la capa lineal (x.contiguous().view(-1, x.size(-1))) 
    y se pasa a través de la capa lineal self.fc. Finalmente, se remodela la 
    salida y para que tenga la forma (batch_size, sequence_length, n_out) antes 
    de retornarla.

"""

class DeepRNNIC(torch.nn.Module):
  def __init__(self, n_out=10, dropout=0):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=2, batch_first=True, dropout=dropout)
    self.fc = torch.nn.Linear(20, n_out)

  def forward(self, x):
    x, h = self.rnn(x) 
    x_reshaped = x.contiguous().view(-1, x.size(-1))
    y = self.fc(x_reshaped)
    y = y.contiguous().view(x.size(0), -1, y.size(-1))
    return y