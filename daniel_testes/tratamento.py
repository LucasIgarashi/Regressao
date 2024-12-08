import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Carregar os dados
minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

#formatando cabeçalhos 
minha_base.columns = minha_base.columns.str.strip()  # Remove espaços no início/fim
minha_base.columns = minha_base.columns.str.lower()  # Converte para minúsculas 

# ========================================
# TRATAMENTO DE DADOS INDESEJADOS
# ========================================

# Salvando Precos
precos = minha_base["preco"]

# Retirando linhas NA
minha_base = minha_base.dropna()

# Retirando colunas indesejadas
atributos = minha_base.drop(columns=['id', 'radio_am_fm', 'data_ultima_lavagem', 'volume_motor', 'modelo', 'débitos', 'preco', 'portas', 'tração'])
minha_base = atributos 

#KM: tem km escrito
minha_base["km"] = minha_base["km"].str.replace(" km", "", regex=False) # Removendo o texto 'km' e os espaços associados da coluna
minha_base["km"] = minha_base["km"].astype(float) # Convertendo os valores para números float

# ========================================
# TRATAMENTO DE DADOS CATEGÓRICOS
# ========================================


# Transformando dados categóricos ORDINARIOS em números
# faixa_preco
ordem = ["Econômico", "Médio", "Luxo", "Muito Luxo"]
encoder = OrdinalEncoder(categories=[ordem])
# minha_base['faixa_preco'] = encoder.fit_transform(minha_base[['faixa_preco']])


# Transformando dados categóricos DISCRETAS e NÃO Ordinarias em números
# classificacao_veiculo, codigo_concessionaria, adesivos_personalizados, cor, tipo_cambio, combustivel,couro,categoria,ano,fabricante
# minha_base.to_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/base_sem_dumie.csv", index=False)
minha_base = pd.get_dummies(minha_base, dtype=float)

# ========================================
# PADRONIZAÇÃO
# ========================================
padronizar = StandardScaler()
padronizar.fit(minha_base)
atributos_padronizados = padronizar.transform(minha_base)

# ========================================
# SALVANDO A BASE TRATADA
# ========================================
# Salvar como CSV
# minha_base.to_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/base_tratada.csv", index=False)

