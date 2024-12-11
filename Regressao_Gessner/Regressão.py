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


# Pré-processamento da base de dados treino
def Processar_Dados_Treino(dados):
    # Exclui as colunas inúteis
    dados.drop(dados.columns[[0,1,2,3, 5,6,8,11,12,13,14,15,16,18,19,20,21,22,23]], axis=1, inplace=True)

    # Arruma a sintaxe dos valores para o Combustível
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
    dados["Km"] = dados["Km"].astype(float)

    # Preenche as células de KM e PRECO com a média
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())

    # Exclui as linhas que possui alguma das celulas vazias
    dados = dados.dropna()

    # Muda variável categórica para contínua
    faixa_preco = dados.iloc[:, -1]
    dados = pd.get_dummies(dados.iloc[:, :-1], drop_first=True, dtype=float)
    dados = pd.concat([dados, faixa_preco], axis=1)
    ordem = ["Econômico", "Médio", "Luxo", "Muito Luxo"]
    encoder = OrdinalEncoder(categories=[ordem])
    dados['Faixa_Preco'] = encoder.fit_transform(dados[['Faixa_Preco']])

    # Separa o PRECO da base de dados 
    precos = dados["Preco"]
    atributos = dados.drop(columns=["Preco"])

    # Padroniza a base de dados
    padronizar = StandardScaler()
    padronizar.fit(atributos)
    atributos_padronizados = padronizar.transform(atributos)

    # Remove os outliers
    tirar_outl = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05)
    tirar_outl.fit(atributos_padronizados)
    amostras_normais = tirar_outl.predict(atributos_padronizados) == 1
    dados_limpos = atributos[amostras_normais]
    precos_limpos = precos[amostras_normais]

    # Converte Pandas -> Numpy
    dados_limpos = dados_limpos.to_numpy()
    precos_limpos = precos_limpos.to_numpy()

    return dados_limpos, precos_limpos

# Pré-processamento da base de dados teste
def Processar_Dados_Teste(dados):
    # Exclui as colunas inúteis
    dados.drop(dados.columns[[0,1,2,3, 5,6,8,11,12,13,14,15,16,18,19,20,21,22,23]], axis=1, inplace=True)

    # Arruma a sintaxe dos valores para o Combustível
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
    dados["Km"] = dados["Km"].astype(float)

    # Preenche as células de KM e PRECO com a média
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())

    # Exclui as linhas que possui alguma das celulas vazias
    dados = dados.dropna()

    # Muda variável categórica para contínua
    faixa_preco = dados.iloc[:, -1]
    dados = pd.get_dummies(dados.iloc[:, :-1], drop_first=True, dtype=float)
    dados = pd.concat([dados, faixa_preco], axis=1)
    ordem = ["Econômico", "Médio", "Luxo", "Muito Luxo"]
    encoder = OrdinalEncoder(categories=[ordem])
    dados['Faixa_Preco'] = encoder.fit_transform(dados[['Faixa_Preco']])

    # Separa o PRECO da base de dados 
    precos = dados["Preco"]
    atributos = dados.drop(columns=["Preco"])

    # Padroniza a base de dados
    padronizar = StandardScaler()
    padronizar.fit(atributos)
    atributos_padronizados = padronizar.transform(atributos)

    return atributos_padronizados, precos

# Plota o gráfico heatmap
def plot_heatmap_scatter(modelo, dados, precos, nome_modelo):
    predicoes = modelo.predict(dados)

    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        x=predicoes,
        y=precos,
        cmap="coolwarm",
        fill=True,
        thresh=0,
        levels=100
    )
    plt.scatter(predicoes, precos, color='black', s=10, alpha=0.5, label='Dados Individuais')
    
    # min_val = min(min(precos), min(predicoes))
    # max_val = max(max(precos), max(predicoes))
    # plt.xlim(min_val, max_val)
    # plt.ylim(min_val, max_val)

    plt.axline((0, 0), (1,1), slope=None, color="pink", linestyle="--", label="y=x (Ideal)")

    plt.title(f"Gráfico de Dispersão - {nome_modelo}")
    plt.xlabel("Preços Previstos")
    plt.ylabel("Preços Reais")
    plt.legend()
    plt.show()



def avaliar_modelo_com_validacao_cruzada(modelo, dados_treino, precos_treino, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    r2_scores = cross_val_score(
        modelo, 
        dados_treino, 
        precos_treino, 
        cv=kf, 
        scoring='r2'
    )
    
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


# Imprime os resultados dos métodos
def imprimir_resultados_validacao(nome_metodo, resultados):
    print(f"MÉTODO: {nome_metodo}")
    print(f"R2 Médio: {resultados['R2_medio']:.4f} (±{resultados['R2_std']:.4f})")
    print(f"MAE Médio: {resultados['MAE_medio']:.4f} (±{resultados['MAE_std']:.4f})")
    print("------------")


# ============================================================


# Carregar os dados
dados_treino, precos_treino = Processar_Dados_Treino(pd.read_csv(r"C:\Users\lucas\OneDrive - MSFT\UFSC\7º Semestre - 24.2\0 - Disciplinas\Sistemas Inteligentes\0 - Importante\Code\Regressao\train.csv"))
dados_teste, precos_teste = Processar_Dados_Teste(pd.read_csv(r"C:\Users\lucas\OneDrive - MSFT\UFSC\7º Semestre - 24.2\0 - Disciplinas\Sistemas Inteligentes\0 - Importante\Code\Regressao\train.csv"))


modelos = [
    ('Regressão Linear', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators=100)),
    ('Gradient Boosting', GradientBoostingRegressor(
        loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0,
        criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
        init=None, random_state=None, max_features=None, alpha=0.9, verbose=0,
        max_leaf_nodes=None, warm_start=False, validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0
    ))
]

for nome, modelo in modelos:
    resultados = avaliar_modelo_com_validacao_cruzada(modelo, dados_treino, precos_treino)
    imprimir_resultados_validacao(nome, resultados)

    # Treinar o modelo e fazer previsões
    modelo.fit(dados_treino, precos_treino)
    plot_heatmap_scatter(modelo, dados_teste, precos_teste, nome)