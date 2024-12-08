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



def Processar_Dados(dados):

    '''EXCLUI COLUNAS INÚTEIS:''' 
    dados.drop(dados.columns[[0,1,2,3, 5,6,8,11,12,13,14,15,16,18,19,20,21,22,23]], axis=1, inplace=True)


    '''ARRUMAR SINTAXE'''

    #Combustível - Gás Natural, Gasolina, Diesel estavam escritos de formas diferentes. Realizando as substituições: 
    dados["Combustivel"] = dados["Combustivel"].replace({
        "Gás Natural": "Gás",
        "gasolina": "Gasolina",
        "Gasol.": "Gasolina",
        "GASOLINA": "Gasolina",
        "DIESEL": "Diesel",
        "diesel": "Diesel",
        "Dies.": "Diesel"
    }) 

    #KM: tem km escrito
    dados["Km"] = dados["Km"].str.replace(" km", "", regex=False) # Removendo o texto 'km' e os espaços associados da coluna
    dados["Km"] = dados["Km"].astype(float) # Convertendo os valores para números float

    print('-----------------------')
    print('Sintaxe Arrumada:')
    print(dados)
    print()


    '''MUDAR VARIÁVEL NOMINAL PARA CONTÍNUA'''
    #One Hot Enconding, cria uma coluna para cada categoria existente, aumenta a dimensionalidade. Usa quando tem poucas categorias, como em Combustivel, Classificacao_Veiculo e Faixa_Preco.
    # Applying one-hot encoding
    dados = pd.get_dummies(dados, dtype=float)

    print('-----------------------')
    print('Nominais viraram atributos:')
    print(dados)
    print()


    '''PREENCHER LACUNAS'''
    #Preencher Km pela média dos Km.
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    #Preencher Km pela média dos Km.
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())

    print('-----------------------')
    print('Lacunas Preenchihas:')
    print(dados)
    print()


    '''SEPARA O PREÇO(target) DOS ATRIBUTOS'''
    precos = dados["Preco"]
    atributos = dados.drop(columns=["Preco"])


    '''PADRONIZA'''
    padronizar = StandardScaler()
    padronizar.fit(atributos)
    atributos_padronizados = padronizar.transform(atributos) #vira np array

    print('-----------------------')
    print('Dados padronizados:')
    print(atributos_padronizados)


    '''TIRA OUTLIERS''' 
    #Tirar o outliers depois de padronizar, pois o método OneClassSVM é sensível à outliers.

    tirar_outl = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)  # Ajustar `gamma` e `nu` conforme necessário
    tirar_outl.fit(atributos_padronizados)
    amostras_normais = tirar_outl.predict(atributos_padronizados) == 1 #O modelo retorna 1 para amostras consideradas normais e -1 para outliers.

    #Filtra os dados dos outliers descobertos
    dados_limpos = atributos[amostras_normais]
    precos_limpos = precos[amostras_normais]


    '''MUDA DE VOLTA PARA NUMPY'''
    dados_limpos = dados_limpos.to_numpy()
    precos_limpos = precos_limpos.to_numpy()

    print('-----------------------')
    print('Dados sem outliers:')
    print(dados_limpos)



    print('-----------------------')
    print('Precos sem outliers:')
    print(precos_limpos)

    return dados_limpos, precos_limpos




'''CARREGAR OS DADOS'''
dados = pd.read_csv(r"C:\Users\Darth\Documents\Sistemas Inteligentes\Regressão\train.csv")

'''SEPARA EM DADOS TESTE E TREINO:'''
dados_treino, dados_teste = train_test_split(dados, train_size=0.0012, random_state=11) 
print('-----------------------')
print('Dados de Treino:')
print(dados_treino)
print()

dados_processados, precos = Processar_Dados(dados_treino)
