import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns

# Função para determinar a faixa de preço com base nos limites dinâmicos
def determinar_faixa_preco(preco, limites):
    if preco <= limites['Econômico']: return "Econômico"
    elif limites['Econômico'] < preco <= limites['Médio']: return "Médio"
    elif limites['Médio'] < preco <= limites['Luxo']: return "Luxo"
    else: return "Muito Luxo"

# Pré-processamento da base de dados treino
def Processar_Dados_Treino(dados):
    # Exclui as colunas inúteis
    dados.drop(dados.columns[[0,1,2,3, 5,6, 8, 11,12,13,14,15,16, 18,19,20,21,22,23]], axis=1, inplace=True)

    # dados = dados.drop(columns=['NOME1', 'NOME2'])
    print(dados.columns.to_list())

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

    # Preenche as células de KM, PRECO com a média
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())
    
    # Calcular os limites dinâmicos para cada FAIXA_PRECO
    limites = {
        'Econômico': dados[dados['Faixa_Preco'] == 'Econômico']['Preco'].max(),
        'Médio': dados[dados['Faixa_Preco'] == 'Médio']['Preco'].max(),
        'Luxo': dados[dados['Faixa_Preco'] == 'Luxo']['Preco'].max(),
        'Muito Luxo': dados[dados['Faixa_Preco'] == 'Muito Luxo']['Preco'].max()
        }

    dados["Faixa_Preco"] = dados["Preco"].apply(determinar_faixa_preco, limites=limites)
    
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

    # Preenche as células de KM, PRECO com a média
    dados["Km"] = dados["Km"].fillna(dados["Km"].mean())
    dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())
    
    # Calcular os limites dinâmicos para cada FAIXA_PRECO
    limites = {
        'Econômico': dados[dados['Faixa_Preco'] == 'Econômico']['Preco'].max(),
        'Médio': dados[dados['Faixa_Preco'] == 'Médio']['Preco'].max(),
        'Luxo': dados[dados['Faixa_Preco'] == 'Luxo']['Preco'].max(),
        'Muito Luxo': dados[dados['Faixa_Preco'] == 'Muito Luxo']['Preco'].max()
        }

    # Preenche a coluna Faixa_Preco com base nos valores de Preco e nos limites dinâmicos
    dados["Faixa_Preco"] = dados["Preco"].apply(determinar_faixa_preco, limites=limites)

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
def plot_heatmap_scatter(modelo, dados, precos):
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

    plt.legend()
    plt.ylabel('')
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
    print("-------------------------------------")
    print(f"MÉTODO: {nome_metodo}")
    print(f"R2 Médio: {resultados['R2_medio']:.4f} (±{resultados['R2_std']:.4f})")
    print(f"MAE Médio: {resultados['MAE_medio']:.4f} (±{resultados['MAE_std']:.4f})")
    print("-------------------------------------")


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
    model = modelo.fit(dados_treino, precos_treino)
    plot_heatmap_scatter(modelo, dados_teste, precos_teste)

# import joblib
# model.dump('caminho.joblib')

# model = joblib.load(with caminho open('rb') as file)