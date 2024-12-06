import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

# Número de amostras únicas na coluna (supondo que a coluna 3 seja "modelos")
numero_modelos = minha_base.iloc[:, 3].nunique()  # Use iloc para acessar a coluna pela posição
print("Número de modelos únicos:", numero_modelos)

numero_fabricas = minha_base.iloc[:, 2].nunique() 
print("Número de fabricas únicas:", numero_fabricas)

numero_categoria = minha_base.iloc[:,5].nunique()
print("Número de categorias únicas:", numero_categoria)
