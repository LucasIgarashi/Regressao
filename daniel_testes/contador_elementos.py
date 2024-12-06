import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# A = 0, B = 1, C = 2, D = 3, E = 4, F = 5, G = 6, H = 7, I = 8, J = 9, K = 10, L = 11, M = 12, N = 13, O = 14, P = 15, Q = 16, R = 17, S = 18, T = 19, U 
# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

# Número de amostras únicas na coluna (supondo que a coluna 3 seja "modelos")
numero_modelos = minha_base.iloc[:, 3].nunique()  # Use iloc para acessar a coluna pela posição
print("Número de modelos únicos:", numero_modelos)

numero_fabricas = minha_base.iloc[:, 2].nunique() 
print("Número de fabricas únicas:", numero_fabricas)

numero_categoria = minha_base.iloc[:,5].nunique()
print("Número de categorias únicas:", numero_categoria)

numero_cambios = minha_base.iloc[:, 11]
print("Número de cambios únicos:", numero_cambios.nunique())

numero_id=minha_base.iloc[:, 0]
print("Número de IDs únicos:", numero_id.nunique())

duplicatas = minha_base[minha_base.duplicated()]
print("\nNúmero de duplicatas:", len(duplicatas))
print("linhas duplicatas:", duplicatas)
