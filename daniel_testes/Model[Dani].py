import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder





def Processar_Dados_Treino(minha_base):

    #formatando cabeçalhos 
    minha_base.columns = minha_base.columns.str.strip()  # Remove espaços no início/fim
    minha_base.columns = minha_base.columns.str.lower()  # Converte para minúsculas 

    # ========================================
    # TRATAMENTO DE DADOS INDESEJADOS
    # ========================================

    # Retirando linhas NA
    minha_base = minha_base.dropna()

    # Salvando Precos
    precos = minha_base["preco"]

    # Retirando colunas indesejadas
    minha_base = minha_base.drop(columns=['id', 'radio_am_fm', 'data_ultima_lavagem', 'volume_motor', 'modelo', 'débitos', 'preco', 'portas', 'tração'])

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
    minha_base['faixa_preco'] = encoder.fit_transform(minha_base[['faixa_preco']])


    # Transformando dados categóricos DISCRETAS e NÃO Ordinarias em números
    # classificacao_veiculo, codigo_concessionaria, adesivos_personalizados, cor, tipo_cambio, combustivel,couro,categoria,ano,fabricante

    minha_base = pd.get_dummies(minha_base, dtype=float)

    # ========================================
    # PADRONIZAÇÃO
    # ========================================
    padronizar = StandardScaler()
    padronizar.fit(minha_base)
    atributos_padronizados = padronizar.transform(minha_base)

    return atributos_padronizados, precos



def Processar_Dados_Teste(minha_base):

 
        #formatando cabeçalhos 
    minha_base.columns = minha_base.columns.str.strip()  # Remove espaços no início/fim
    minha_base.columns = minha_base.columns.str.lower()  # Converte para minúsculas 

    # ========================================
    # TRATAMENTO DE DADOS INDESEJADOS
    # ========================================

    # Retirando linhas NA
    minha_base = minha_base.dropna()

    # Salvando Precos
    precos = minha_base["preco"]

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


    # Transformando dados categóricos DISCRETAS e N    #formatando cabeçalhos 
    minha_base.columns = minha_base.columns.str.strip()  # Remove espaços no início/fim
    minha_base.columns = minha_base.columns.str.lower()  # Converte para minúsculas 
    
    # ========================================
    # TRATAMENTO DE DADOS INDESEJADOS
    # ========================================
    
    # Retirando linhas NA
    minha_base = minha_base.dropna()
    
    # Salvando Precos
    precos = minha_base["preco"]
    
    # Retirando colunas indesejadas
    minha_base = minha_base.drop(columns=['id', 'radio_am_fm', 'data_ultima_lavagem', 'volume_motor', 'modelo', 'débitos', 'preco', 'portas', 'tração'])
    
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
    minha_base['faixa_preco'] = encoder.fit_transform(minha_base[['faixa_preco']])
    
    
    # Transformando dados categóricos DISCRETAS e NÃO Ordinarias em números
    # classificacao_veiculo, codigo_concessionaria, adesivos_personalizados, cor, tipo_cambio, combustivel,couro,categoria,ano,fabricante

    minha_base = pd.get_dummies(minha_base, dtype=float)
    
    # ========================================
    # PADRONIZAÇÃO
    # ========================================
    padronizar = StandardScaler()
    padronizar.fit(minha_base)
    atributos_padronizados = padronizar.transform(minha_base)

    return atributos_padronizados, precos



'''MÉTODO: RANDOM FOREST'''
def Random_Forest(dados_treino, precos_treino, dados_teste):
    
    amazonia = RandomForestRegressor(n_estimators=100)
    amazonia.fit(dados_treino, precos_treino)

    #Previsão
    precos_previstos = amazonia.predict(dados_teste)

    return precos_previstos


'''MÉTODO: SVR'''
def Svr(dados_treino, precos_treino, dados_teste):
    srv = SVR()
    srv.fit(dados_treino, precos_treino)

    #Previsão
    precos_previstos = srv.predict(dados_teste)

    return precos_previstos


'''MÉTODO: REGRESSÃO LINEAR'''
def Regressao_Linear(dados_treino, precos_treino, dados_teste):
    linear = LinearRegression()
    linear.fit(dados_treino, precos_treino)

    #Previsão
    precos_previstos = linear.predict(dados_teste)
    
    return precos_previstos


'''R2 Ajustado''' #0 à 1
def R2(labels_test, labels_pred,data_train):

    adj_r2 = (1 - ((1 - r2_score(labels_test, labels_pred)) * (len(labels_test) - 1)) / 
            (len(labels_test) - data_train.shape[1] - 1))

    return adj_r2





'''CARREGAR OS DADOS'''
dados = pd.read_csv(r"C:\Users\Darth\Documents\Sistemas Inteligentes\Regressão\train.csv")


'''SEPARA EM DADOS TESTE E TREINO:'''
cru_treino, cru_teste = train_test_split(dados, train_size=0.5, random_state=7) 


'''PROCESSA DADOS'''
dados_treino, precos_treino = Processar_Dados_Treino(cru_treino)
dados_teste, precos_teste = Processar_Dados_Teste(cru_teste)


'''MÉTODO: REGRESSÃO LINEAR'''
precos_previstos = Regressao_Linear(dados_treino, precos_treino, dados_teste)
print("MÉTODO: REGRESSÃO LINEAR")
print("R2",R2(precos_teste, precos_previstos, dados_treino))
MAE = mean_absolute_error(precos_teste, precos_previstos)
print("MAE",MAE)  
print("------------") 


'''MÉTODO: SVR'''
precos_previstos = Svr(dados_treino, precos_treino, dados_teste)
print("MÉTODO: SVR")
print("R2",R2(precos_teste, precos_previstos, dados_treino))
MAE = mean_absolute_error(precos_teste, precos_previstos)
print("MAE",MAE)   
print("------------") 


'''MÉTODO: RANDOM FOREST'''
precos_previstos = Random_Forest(dados_treino, precos_treino, dados_teste)
print("MÉTODO: RANDOM FOREST")
print("R2",R2(precos_teste, precos_previstos, dados_treino))
MAE = mean_absolute_error(precos_teste, precos_previstos)
print("MAE",MAE) 
print("------------") 


