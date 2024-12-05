import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

# Ajustando colunas (Preço)
colunas = list(minha_base.columns)  # Cria lista de colunas
colunas[-1], colunas[17] = colunas[17], colunas[-1]  # Troca última coluna com a coluna 18
minha_base = minha_base[colunas]

# Separando rótulos e dados
data = np.array(minha_base.iloc[1:, :-1])  # Ignorar a primeira linha e excluir a última coluna
labels = np.array(minha_base.iloc[1:, -1])  # Ignorar a primeira linha e pegar a última coluna

# Exibindo shapes para verificar
print("Shape dos dados:", data.shape)
print("Shape dos rótulos:", labels.shape)
