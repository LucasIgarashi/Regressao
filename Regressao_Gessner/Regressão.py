
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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns


''' PRÉ-PROCESSAMENTO DE DADOS '''

def Processar_Dados_Treino(dados):

    '''EXCLUI COLUNAS INÚTEIS:''' #Sobram as colunas: Ano, Combustivel, Km, Cilindros, Preco, Classificacao_veiculo, Faixa_Preco 
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

    '''EXCLUI COLUNAS INÚTEIS:''' #Sobram as colunas: Ano, Combustivel, Km, Cilindros, Preco, Classificacao_veiculo, Faixa_Preco 
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





'''VALIDAÇÂO CRUZADA'''

def avaliar_modelo_com_validacao_cruzada(modelo, dados_treino, precos_treino, n_splits=5):
    """
    Realiza validação cruzada para um modelo de regressão
    
    Parâmetros:
    - modelo: Modelo de regressão a ser avaliado
    - dados_treino: Características de treino
    - precos_treino: Valores alvo de treino
    - n_splits: Número de divisões para validação cruzada
    
    Retorna:
    - Média dos scores R²
    - Média dos MAEs
    """
    # Configuração da validação cruzada
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calcula os scores de R²
    r2_scores = cross_val_score(
        modelo, 
        dados_treino, 
        precos_treino, 
        cv=kf, 
        scoring='r2'
    )
    
    # Calcula os MAEs (usando negative_mean_absolute_error)
    mae_scores = -cross_val_score(
        modelo, 
        dados_treino, 
        precos_treino, 
        cv=kf, 
        scoring='neg_mean_absolute_error'
    )
    
    return {
        'R2_medio': r2_scores.mean(),
        'R2_std': r2_scores.std(),
        'MAE_medio': mae_scores.mean(),
        'MAE_std': mae_scores.std()
    }


# Função para imprimir resultados da validação cruzada
def imprimir_resultados_validacao(nome_metodo, resultados):
    print(f"MÉTODO: {nome_metodo}")
    print(f"R2 Médio: {resultados['R2_medio']:.4f} (±{resultados['R2_std']:.4f})")
    print(f"MAE Médio: {resultados['MAE_medio']:.4f} (±{resultados['MAE_std']:.4f})")
    print("------------")





'''CARREGAR OS DADOS'''
#dados = pd.read_csv(r"C:\Users\Darth\Documents\Sistemas Inteligentes\Regressão\train.csv")

'''PROCESSA DADOS'''
dados_treino, precos_treino = Processar_Dados_Treino(pd.read_csv(r"C:\Users\Darth\Documents\Sistemas Inteligentes\Regressão\train.csv"))
dados_teste, precos_teste = Processar_Dados_Teste(pd.read_csv(r"C:\Users\Darth\Documents\Sistemas Inteligentes\Regressão\train.csv"))
#dados_teste = dados_treino
#precos_teste = precos_treino


'''AVALIAR OS MODELOS'''
# Modelos a serem avaliados
modelos = [
    ('Regressão Linear', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators=100)),
    ('Gradient', GradientBoostingRegressor( loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0))

]


# Realizando validação cruzada para cada modelo
print("==================================\nDados Treino")
for nome, modelo in modelos:
    resultados = avaliar_modelo_com_validacao_cruzada(modelo, dados_treino, precos_treino)
    imprimir_resultados_validacao(nome, resultados)
print("==================================")


# Gerar os preços previstos e reais, e criar heatmaps
for nome, modelo in modelos:
    # Treinar o modelo e fazer previsões
    modelo.fit(dados_treino, precos_treino)

    predicoes = modelo.predict(dados_teste)




    # Criar o heatmap scatter
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        x=predicoes,
        y=precos_teste,
        cmap="coolwarm",  # Azul (menos pontos) para vermelho (mais pontos)
        fill=True,
        thresh=0,         # Define que áreas sem densidade aparecem brancas
        levels=100        # Níveis de contorno para densidade
    )

    # Adicionar a reta bissetriz
    plt.axline((0, 0), (1,1),slope=None, color="pink", linestyle="--", label="y=x (Ideal)")

    # Personalizar o gráfico
    plt.title(f"Heatmap Scatter - {nome}")
    plt.xlabel("Preços Previstos")
    plt.ylabel("Preços Reais")
    plt.legend()
    plt.show()




''' PRÉ-PROCESSAMENTO DE DADOS '''

def Processar_Dados_Treino(dados):

    '''EXCLUI COLUNAS INÚTEIS:''' #Sobram as colunas: Ano, Combustivel, Km, Cilindros, Preco, Classificacao_veiculo, Faixa_Preco 
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

    '''EXCLUI COLUNAS INÚTEIS:''' #Sobram as colunas: Ano, Combustivel, Km, Cilindros, Preco, Classificacao_veiculo, Faixa_Preco 
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





'''VALIDAÇÂO CRUZADA'''

def avaliar_modelo_com_validacao_cruzada(modelo, dados_treino, precos_treino, n_splits=5):
    """
    Realiza validação cruzada para um modelo de regressão
    
    Parâmetros:
    - modelo: Modelo de regressão a ser avaliado
    - dados_treino: Características de treino
    - precos_treino: Valores alvo de treino
    - n_splits: Número de divisões para validação cruzada
    
    Retorna:
    - Média dos scores R²
    - Média dos MAEs
    """
    # Configuração da validação cruzada
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calcula os scores de R²
    r2_scores = cross_val_score(
        modelo, 
        dados_treino, 
        precos_treino, 
        cv=kf, 
        scoring='r2'
    )
    
    # Calcula os MAEs (usando negative_mean_absolute_error)
    mae_scores = -cross_val_score(
        modelo, 
        dados_treino, 
        precos_treino, 
        cv=kf, 
        scoring='neg_mean_absolute_error'
    )
    
    return {
        'R2_medio': r2_scores.mean(),
        'R2_std': r2_scores.std(),
        'MAE_medio': mae_scores.mean(),
        'MAE_std': mae_scores.std()
    }


# Função para imprimir resultados da validação cruzada
def imprimir_resultados_validacao(nome_metodo, resultados):
    print(f"MÉTODO: {nome_metodo}")
    print(f"R2 Médio: {resultados['R2_medio']:.4f} (±{resultados['R2_std']:.4f})")
    print(f"MAE Médio: {resultados['MAE_medio']:.4f} (±{resultados['MAE_std']:.4f})")
    print("------------")





'''CARREGAR OS DADOS'''
#dados = pd.read_csv(r"C:\Users\Darth\Documents\Sistemas Inteligentes\Regressão\train.csv")

'''PROCESSA DADOS'''
dados_treino, precos_treino = Processar_Dados_Teste(pd.read_csv(r"C:\Users\Darth\Documents\Sistemas Inteligentes\Regressão\train.csv"))
dados_teste, precos_teste = Processar_Dados_Teste(pd.read_csv(r"C:\Users\Darth\Documents\Sistemas Inteligentes\Regressão\train.csv"))
#dados_teste = dados_treino
#precos_teste = precos_treino


'''AVALIAR OS MODELOS'''
# Modelos a serem avaliados
modelos = [
    ('Regressão Linear', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators=100)),
    ('Gradient', GradientBoostingRegressor( loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0))

]


# Realizando validação cruzada para cada modelo
for nome, modelo in modelos:
    resultados = avaliar_modelo_com_validacao_cruzada(modelo, dados_treino, precos_treino)
    imprimir_resultados_validacao(nome, resultados)


# Gerar os preços previstos e reais
for nome, modelo in modelos:
    modelo.fit(dados_treino, precos_treino)
    predicoes = modelo.predict(dados_teste)

    # Criar o heatmap scatter
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        x=predicoes,
        y=precos_teste,
        cmap="coolwarm",
        fill=True,
        thresh=0,
        levels=100
    )
    plt.title(f"Heatmap Scatter - {nome}")
    plt.ylabel("Preços Reais")
    plt.xlabel("Preços Previstos")
    plt.show()

