import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

#formatando cabeçalhos 
minha_base.columns = minha_base.columns.str.strip()  # Remove espaços no início/fim
minha_base.columns = minha_base.columns.str.lower()  # Converte para minúsculas (opcional)

# ========================================
# TRATAMENTO DE DADOS INDESEJADOS
# ========================================

# Salvando Precos
precos = minha_base["preco"]

# Retirando linhas NA
minha_base = minha_base.dropna()

# Retirando colunas indesejadas
atributos = minha_base.drop(columns=['id', 'radio_am_fm', 'data_ultima_lavagem', 'volume_motor', 'modelo', 'débitos', 'preco'])
minha_base = atributos 

# ========================================
# TRATAMENTO DE DADOS CATEGÓRICOS
# ========================================


# Transformando dados categóricos em números
ordem = ["Econômico", "Médio", "Luxo", "Muito Luxo"]
encoder = OrdinalEncoder(categories=[ordem])
minha_base['faixa_preco'] = encoder.fit_transform(minha_base[['faixa_preco']])

# ========================================
# SALVANDO A BASE TRATADA
# ========================================
# Salvar como CSV
minha_base.to_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/base_tratada.csv", index=False)

print("Base tratada salva com sucesso!")
