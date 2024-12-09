import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor




def Processar_Dados_Treino(dados):

    '''EXCLUI COLUNAS INÚTEIS:''' #Ano, Combustivel, Km, Cilindros, Preco, Faixa_Preco 
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


    '''PREENCHER LACUNAS'''
    #Preencher Km pela média dos Km.
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    #Preencher Km pela média dos Km.
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())
    

    '''EXCLUI LINHAS COM DADOS FALTANTES'''
    # Removendo linhas com valores ausentes
    dados = dados.dropna()


    '''MUDAR VARIÁVEL NOMINAL PARA CONTÍNUA'''
    #One Hot Enconding, cria uma coluna para cada categoria existente, aumenta a dimensionalidade. Usa quando tem poucas categorias, como em Combustivel, Classificacao_Veiculo e Faixa_Preco.
    faixa_preco = dados.iloc[:, -1]  # Seleciona a última coluna
    dados = pd.get_dummies(dados.iloc[:, :-1],drop_first=True, dtype=float)
    dados = pd.concat([dados, faixa_preco], axis=1)


    # Transformando dados categóricos ORDINARIOS em números (faixa_preco)
    ordem = ["Econômico", "Médio", "Luxo", "Muito Luxo"]
    encoder = OrdinalEncoder(categories=[ordem])
    dados['Faixa_Preco'] = encoder.fit_transform(dados[['Faixa_Preco']])
    

    '''SEPARA O PREÇO(target) DOS ATRIBUTOS'''
    precos = dados["Preco"]
    atributos = dados.drop(columns=["Preco"])


    '''PADRONIZA'''
    padronizar = StandardScaler()
    padronizar.fit(atributos)
    atributos_padronizados = padronizar.transform(atributos) #vira np array


    '''TIRA OUTLIERS''' 
    #Tirar o outliers depois de padronizar, pois o método OneClassSVM é sensível à outliers.

    tirar_outl = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05) 
    tirar_outl.fit(atributos_padronizados)
    amostras_normais = tirar_outl.predict(atributos_padronizados) == 1 #O modelo retorna 1 para amostras consideradas normais e -1 para outliers.

    #Filtra os dados dos outliers descobertos
    dados_limpos = atributos[amostras_normais]
    precos_limpos = precos[amostras_normais]


    '''MUDA DE VOLTA PARA NUMPY'''
    dados_limpos = dados_limpos.to_numpy()
    precos_limpos = precos_limpos.to_numpy()


    return dados_limpos, precos_limpos



def Processar_Dados_Teste(dados):

    '''EXCLUI COLUNAS INÚTEIS:''' #Ano, Combustivel, Km, Cilindros, Preco, Classificacao_veiculo, Faixa_Preco 
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


    '''PREENCHER LACUNAS'''
    #Preencher Km pela média dos Km.
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    #Preencher Km pela média dos Km.
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())
    

    '''EXCLUI LINHAS COM DADOS FALTANTES'''
    # Removendo linhas com valores ausentes
    dados = dados.dropna()


    '''MUDAR VARIÁVEL NOMINAL PARA CONTÍNUA'''
    #One Hot Enconding, cria uma coluna para cada categoria existente, aumenta a dimensionalidade. Usa quando tem poucas categorias, como em Combustivel, Classificacao_Veiculo e Faixa_Preco.
    faixa_preco = dados.iloc[:, -1]  # Seleciona a última coluna
    dados = pd.get_dummies(dados.iloc[:, :-1],drop_first=True, dtype=float)
    dados = pd.concat([dados, faixa_preco], axis=1)


    # Transformando dados categóricos ORDINARIOS em números (faixa_preco)
    ordem = ["Econômico", "Médio", "Luxo", "Muito Luxo"]
    encoder = OrdinalEncoder(categories=[ordem])
    dados['Faixa_Preco'] = encoder.fit_transform(dados[['Faixa_Preco']])
    

    '''SEPARA O PREÇO(target) DOS ATRIBUTOS'''
    precos = dados["Preco"]
    atributos = dados.drop(columns=["Preco"])


    '''PADRONIZA'''
    padronizar = StandardScaler()
    padronizar.fit(atributos)
    atributos_padronizados = padronizar.transform(atributos) #vira np array


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
    srv = SVR( kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    srv.fit(dados_treino, precos_treino)

    #Previsão
    precos_previstos = srv.predict(dados_teste)

    return precos_previstos



def Gradient(dados_treino, precos_treino, dados_teste):
    # Create a lasso regression model
    model = GradientBoostingRegressor( loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
    model.fit(dados_treino, precos_treino)

    # Predict the response for a new data point
    return model.predict(dados_teste)

'''R2 Ajustado''' #0 à 1
def R2(labels_test, labels_pred,data_train):

    adj_r2 = (1 - ((1 - r2_score(labels_test, labels_pred)) * (len(labels_test) - 1)) / 
            (len(labels_test) - data_train.shape[1] - 1))

    return adj_r2

'''R2 Ajustado''' #0 à 1
def R(labels_test, labels_pred):
    return r2_score(labels_test, labels_pred)




   



'''CARREGAR OS DADOS'''
dados = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")


'''SEPARA EM DADOS TESTE E TREINO:'''
cru_treino, cru_teste = train_test_split(dados, train_size=0.5, random_state=7) 


'''PROCESSA DADOS'''
dados_treino, precos_treino = Processar_Dados_Treino(cru_treino)
dados_teste, precos_teste = Processar_Dados_Teste(cru_teste)



'''MÉTODO: SVR'''
precos_previstos = Svr(dados_treino, precos_treino, dados_teste)
print("MÉTODO: SVR")
print("R2",R2(precos_teste, precos_previstos, dados_treino))
print("R",R(precos_teste, precos_previstos))
MAE = mean_absolute_error(precos_teste, precos_previstos)
print("MAE",MAE)   
print("------------") 


'''MÉTODO: RANDOM FOREST'''
precos_previstos = Random_Forest(dados_treino, precos_treino, dados_teste)
print("MÉTODO: RANDOM FOREST")
print("R2",R2(precos_teste, precos_previstos, dados_treino))
print("R",R(precos_teste, precos_previstos))
MAE = mean_absolute_error(precos_teste, precos_previstos)
print("MAE",MAE) 
print("------------") 



'''MÉTODO: GRADIENT'''
precos_previstos = Gradient(dados_treino, precos_treino, dados_teste)
print("MÉTODO: Gradient")
print("R2",R2(precos_teste, precos_previstos, dados_treino))
MAE = mean_absolute_error(precos_teste, precos_previstos)
print("MAE",MAE) 
print("------------") 




