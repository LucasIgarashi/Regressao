import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

# Ajustando colunas (Preço)
colunas = list(minha_base.columns)  # Cria lista de colunas
colunas[-1], colunas[17] = colunas[17], colunas[-1]  # Troca última coluna com a coluna 18
minha_base = minha_base[colunas]

# Separando rótulos e dados
data = np.array(minha_base.iloc[1:, :-1])  # Ignorar a primeira linha e excluir a última coluna
labels = np.array(minha_base.iloc[1:, -1])  # Ignorar a primeira linha e pegar a última coluna

transformador = LabelEncoder() # iniciando obj transformador

# 1A, 2B, 3C, 4D, 5E, 6F, 7G, 8H, 9I, 10J, 11K, 12L, 13M, 14N, 15O, 16P, 17Q, 18R, 19S, 20T, 21U, 22V, 23W, 24X, 25Y, 26Z

#Transfomando colunas 

for i in range(3,5): #3>= i < 5, CD
    data[:, i] = transformador.fit_transform(data[:, i])
    
for i in range(6,9): #6>= i < 9 , FGH
    data[:, i] = transformador.fit_transform(data[:, i])

# 8H
data[:, 8] = transformador.fit_transform(data[:, 8])


# Exibindo shapes para verificar
print("Shape dos dados:", data.shape)
print("Shape dos rótulos:", labels.shape)
