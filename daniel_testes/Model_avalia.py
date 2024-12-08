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




def Processar_Dados_Treino(dados):

    '''EXCLUI COLUNAS INÚTEIS:''' 
    dados.drop(dados.columns[[0,1,2,3, 5,6,8,11,12,13,14,15,16,18,19,20,21,22,23]], axis=1, inplace=True)
    # Eliminamos muitas colunas então podemos ter decidido errado sobre a importancia delas pesquisei dois métodos pra dar uma olhada seleção de características ou análise de variância (ANOVA)

    '''ARRUMAR SINTAXE'''
    dados["Combustivel"] = dados["Combustivel"].replace({
        "Gás Natural": "Gás",
        "gasolina": "Gasolina",
        "Gasol.": "Gasolina",
        "GASOLINA": "Gasolina",
        "DIESEL": "Diesel",
        "diesel": "Diesel",
        "Dies.": "Diesel"
    }) 

    """KM tem km escrito"""
    dados["Km"] = dados["Km"].str.replace(" km", "", regex=False) 


    '''MUDAR VARIÁVEL NOMINAL PARA CONTÍNUA'''
    dados = pd.get_dummies(dados, dtype=float)


    '''PREENCHER LACUNAS'''
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())


    '''SEPARA O PREÇO(target) DOS ATRIBUTOS'''
    precos = dados["Preco"]
    atributos = dados.drop(columns=["Preco"])


    '''PADRONIZA'''
    padronizar = StandardScaler()
    padronizar.fit(atributos)
    atributos_padronizados = padronizar.transform(atributos) #vira np array


    '''TIRA OUTLIERS''' 
    #Nessa parte podemos talvez reavaliar oq estamso tirando de outlier já que podemos ter um outlier verdadeiro que contribui pro nosso modelo
    tirar_outl = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
    tirar_outl.fit(atributos_padronizados)
    amostras_normais = tirar_outl.predict(atributos_padronizados) == 1 

    dados_limpos = atributos[amostras_normais]
    precos_limpos = precos[amostras_normais]


    '''MUDA DE VOLTA PARA NUMPY'''
    dados_limpos = dados_limpos.to_numpy()
    precos_limpos = precos_limpos.to_numpy()

    return dados_limpos, precos_limpos



def Processar_Dados_Teste(dados):

    '''EXCLUI COLUNAS INÚTEIS:''' 
    dados.drop(dados.columns[[0,1,2,3, 5,6,8,11,12,13,14,15,16,18,19,20,21,22,23]], axis=1, inplace=True)


    '''ARRUMAR SINTAXE'''

    dados["Combustivel"] = dados["Combustivel"].replace({
        "Gás Natural": "Gás",
        "gasolina": "Gasolina",
        "Gasol.": "Gasolina",
        "GASOLINA": "Gasolina",
        "DIESEL": "Diesel",
        "diesel": "Diesel",
        "Dies.": "Diesel"
    }) 

    dados["Km"] = dados["Km"].str.replace(" km", "", regex=False) 
    dados["Km"] = dados["Km"].astype(float) #????

        #tomar cuidado se tiver mais info em uma das duas, exemplo tem tudo no trieno mas falta gasosa no teste e vice versa

    '''MUDAR VARIÁVEL NOMINAL PARA CONTÍNUA'''
    dados = pd.get_dummies(dados, dtype=float)


    '''PREENCHER LACUNAS'''
    #tem que discutir isso aqui  ass: @Gessner-Lucas @LucasIgarashi
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())


    '''SEPARA O PREÇO(target) DOS ATRIBUTOS'''
    precos = dados["Preco"]
    atributos = dados.drop(columns=["Preco"])


    '''PADRONIZA'''
    padronizar = StandardScaler()
    padronizar.fit(atributos)
    atributos_padronizados = padronizar.transform(atributos)

    '''MUDA DE VOLTA PARA NUMPY'''
    precos = precos.to_numpy()

    return atributos_padronizados, precos



'''MÉTODO: RANDOM FOREST'''
def Random_Forest(dados_treino, precos_treino, dados_teste):
    
    amazonia = RandomForestRegressor(n_estimators=100)
    amazonia.fit(dados_treino, precos_treino)

    precos_previstos = amazonia.predict(dados_teste)

    return precos_previstos


'''MÉTODO: SVR'''
def Svr(dados_treino, precos_treino, dados_teste):
    srv = SVR()
    srv.fit(dados_treino, precos_treino)

    #O R2R2 negativo ocorre quando o modelo é incapaz de capturar padrões, tendo desempenho pior do que simplesmente prever a média.
    precos_previstos = srv.predict(dados_teste)

    return precos_previstos


'''MÉTODO: REGRESSÃO LINEAR'''
def Regressao_Linear(dados_treino, precos_treino, dados_teste):
    linear = LinearRegression()
    linear.fit(dados_treino, precos_treino)

    
    precos_previstos = linear.predict(dados_teste)
    
    return precos_previstos


'''R2 Ajustado''' #0 à 1
def R2(labels_test, labels_pred,data_train):

    adj_r2 = (1 - ((1 - r2_score(labels_test, labels_pred)) * (len(labels_test) - 1)) / 
            (len(labels_test) - data_train.shape[1] - 1))

    return adj_r2






'''CARREGAR OS DADOS'''
dados = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")


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


"""
reindex

Adiciona colunas ausentes no conjunto de teste: Se Combustivel_Gás está no treino, mas não no teste,
ele adiciona essa coluna ao teste com valores 0 (porque "Gás" não apareceu no teste).

# Realiza o encoding no conjunto de treino
dados_treino = pd.get_dummies(cru_treino, dtype=float)

# Realiza o encoding no conjunto de teste
dados_teste = pd.get_dummies(cru_teste, dtype=float)

# Reindexa o teste para alinhar as colunas com o treino
dados_teste = dados_teste.reindex(columns=dados_treino.columns, fill_value=0)

"""