import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

# Ajustando colunas (Preço)
colunas = list(minha_base.columns)  # Cria lista de colunas
colunas[-1], colunas[17] = colunas[17], colunas[-1]  # Troca última coluna com a coluna 18
minha_base = minha_base[colunas]  # Reordena as colunas no DataFrame

# Separando rótulos e dados
data = np.array(minha_base.iloc[1:, :-1])  # Ignora a primeira linha e exclui a última coluna
labels = np.array(minha_base.iloc[1:, -1])  # Ignora a primeira linha e pega a última coluna

# Inicializando o LabelEncoder
transformador = LabelEncoder()

# Transformando as colunas de 3 a 4 (CD)
for i in range(3, 5):
    data[:, i] = transformador.fit_transform(data[:, i])

# Transformando as colunas de 6 a 8 (FGH)
for i in range(6, 9):
    data[:, i] = transformador.fit_transform(data[:, i])

# Transformando a coluna 8 (H)
data[:, 8] = transformador.fit_transform(data[:, 8])


