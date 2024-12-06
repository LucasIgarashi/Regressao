import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

# Número de amostras únicas na coluna (supondo que a coluna 3 seja "modelos")
numero_modelos = minha_base.iloc[:, 3].nunique()  # Use iloc para acessar a coluna pela posição
print("Número de modelos únicos:", numero_modelos)

numero_fabricas = minha_base.iloc[:, 2].nunique() 
print("Número de fábricas únicas:", numero_fabricas)

numero_categoria = minha_base.iloc[:, 5].nunique()
print("Número de categorias únicas:", numero_categoria)

numero_cambios = minha_base.iloc[:, 11]
print("Número de câmbios únicos:", numero_cambios.nunique())

numero_id = minha_base.iloc[:, 0]
print("Número de IDs únicos:", numero_id.nunique())

# Identificar e contar as duplicatas
duplicatas = minha_base[minha_base.duplicated()]
print("\nNúmero de linhas com todas carc. duplicadas:", len(duplicatas))

# Número total de linhas e colunas
numero_linhas = len(minha_base)
numero_columns = len(minha_base.columns)
print("Número total de linhas:", numero_linhas)
print("Número total de colunas:", numero_columns)

# Diferença entre o número de linhas e o número de IDs únicos
print("Diferença NumL - NumID:\n", numero_linhas - numero_id.nunique())

duplicados = minha_base[minha_base.duplicated(subset='ID', keep=False)]
# Agrupar as linhas com IDs duplicados
id_duplicados = duplicados.groupby('ID').apply(lambda group: group.index.tolist())

# Exibir as linhas duplicadas de forma desejada
cont = 0
for id_valor, indices in id_duplicados.items():
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            print(f"Linha {indices[i]} e Linha {indices[j]}")
            cont += 1
print("numero de duplicados:",cont)
