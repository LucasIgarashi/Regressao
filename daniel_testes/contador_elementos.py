import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

# Número de amostras únicas na coluna (supondo que a coluna 3 seja "modelos")
numero_unico = minha_base.iloc[:, 2].nunique()  # Use iloc para acessar a coluna pela posição
print("Número de modelos únicos:", numero_unico)