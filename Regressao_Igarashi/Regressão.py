import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
import joblib


# ============================
# DEFINIÇÃO DAS FUNÇÕES
# ============================


# Pré-processamento da base de dados
def pre_processamento(dados):

    # Separa o PRECO da base de dados 
    precos = dados["Preco"]
    atributos = dados.drop(columns=["Preco"])

    # Padroniza a base de dados
    padronizar = StandardScaler()
    padronizar.fit(atributos)
    atributos_padronizados = padronizar.transform(atributos)

    return atributos_padronizados, precos

# Salva o modelo d treinamento para não ter que sempre treinar 
def salvar_modelo(modelo, caminho):
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em {caminho}")

#
def carregar_modelo(caminho):
    modelo = joblib.load(caminho)
    print(f"Modelo carregado de {caminho}")
    return modelo

# Função para determinar a faixa de preço com base nos limites dinâmicos
def determinar_faixa_preco(preco, limites):
    if preco <= limites['Econômico']: return "Econômico"
    elif limites['Econômico'] < preco <= limites['Médio']: return "Médio"
    elif limites['Médio'] < preco <= limites['Luxo']: return "Luxo"
    else: return "Muito Luxo"

def tratar_base_dados(caminho_arquivo, limite=10):
    # Carregar os dados 
    dados = pd.read_csv(caminho_arquivo)

    # Exclui as colunas inúteis
    colunas_para_excluir=[
        'ID',
        # 'Débitos',
        # 'Fabricante',
        'Modelo',
        # 'Ano',
        # 'Categoria',
        # 'Couro',
        # 'Combustivel',
        'Volume_motor',
        # 'Km',
        # 'Cilindros',
        'Tipo_cambio',
        # 'Tração',
        'Portas',
        'Rodas',
        'Cor',
        'Airbags',
        # 'Preco',
        # 'Numero_proprietarios',
        'Data_ultima_lavagem',
        # 'Adesivos_personalizados',
        'Radio_AM_FM',
        'Historico_troca_oleo',
        'Codigo_concessionaria',
        # 'Classificacao_Veiculo',
        # 'Faixa_Preco'
    ]
    colunas_existentes = [coluna for coluna in colunas_para_excluir if coluna in dados.columns]
    dados = dados.drop(columns=colunas_existentes)
    print(dados.columns.to_list())

    # Verificar e corrigir a sintaxe dos valores para o Combustível
    if "Combustivel" in dados.columns:
        dados["Combustivel"] = dados["Combustivel"].replace({
            "Gás Natural": "Gás",
            "gasolina": "Gasolina",
            "Gasol.": "Gasolina",
            "GASOLINA": "Gasolina",
            "DIESEL": "Diesel",
            "diesel": "Diesel",
            "Dies.": "Diesel"
        }) 

    # Verificar e corrigir a variável 'Débitos'
    if "Débitos" in dados.columns:
        dados["Débitos"] = dados["Débitos"].replace("-", 0).astype(float)

    # Verificar e corrigir a sintaxe dos valores para o Km
    if "Km" in dados.columns:
        dados["Km"] = dados["Km"].str.replace(" km", "", regex=False)
        dados["Km"] = dados["Km"].astype(float)
        dados["Km"] = dados["Km"].fillna(dados["Km"].mean())

    # # Verificar e preencher as células de PRECO com a média
    # if "Preco" in dados.columns:
    #     dados["Preco"] = dados["Preco"].fillna(dados["Preco"].mean())
    
    # Calcular os limites dinâmicos para cada FAIXA_PRECO
    if "Faixa_Preco" in dados.columns:
        l = {
            'Econômico': dados[dados['Faixa_Preco'] == 'Econômico']['Preco'].max(),
            'Médio': dados[dados['Faixa_Preco'] == 'Médio']['Preco'].max(),
            'Luxo': dados[dados['Faixa_Preco'] == 'Luxo']['Preco'].max(),
            'Muito Luxo': dados[dados['Faixa_Preco'] == 'Muito Luxo']['Preco'].max()
        }
        dados["Faixa_Preco"] = dados["Preco"].apply(determinar_faixa_preco, limites=l)
    
    # Muda variável categórica para contínua
    if "Faixa_Preco" in dados.columns:
        faixa_preco = dados.iloc[:, -1]
        dados = pd.get_dummies(dados.iloc[:, :-1], drop_first=True, dtype=float)
        dados = pd.concat([dados, faixa_preco], axis=1)
        ordem = ["Econômico", "Médio", "Luxo", "Muito Luxo"]
        encoder = OrdinalEncoder(categories=[ordem])
        dados['Faixa_Preco'] = encoder.fit_transform(dados[['Faixa_Preco']])

    # Verificar e converter 'Sim' e 'Não' para valores binários
    if "Adesivos_personalizados" in dados.columns:
        dados["Adesivos_personalizados"] = dados["Adesivos_personalizados"].replace({"Sim": 1, "Não": 0})
    
    # Verificar e converter 'AM', 'FM' e 'AM/FM' para 0, 1 e 2, respectivamente
    if "Radio_AM_FM" in dados.columns:
        dados["Radio_AM_FM"] = dados["Radio_AM_FM"].replace({"AM/FM": 2, "FM": 1, "AM": 0})

    # # Aplicar codificação de frequência para variáveis categóricas com muitas categorias
    # variaveis_categoricas = dados.select_dtypes(include=['object']).columns
    # for coluna in variaveis_categoricas:
    #     num_categorias = dados[coluna].nunique()
    #     if num_categorias > limite:
    #         # frequencias = dados[coluna].value_counts(normalize=True)
    #         # dados[coluna] = dados[coluna].map(frequencias)
    #         dados = dados.drop(columns=[coluna])

    # Excluir as linhas que possuem alguma das células vazias
    dados = dados.dropna()

    # Salvar a base de dados tratada
    dados.to_csv("base_dados_tratada.csv", index=False)
    print("Base de dados tratada salva como 'base_dados_tratada.csv'.")
    print(dados)


def plot_pareto_atributos(dados, n=10):
    # Separar os atributos e o alvo
    atributos = dados.drop(columns=["Preco"])
    alvo = dados["Preco"]

    # Treinar o modelo RandomForestRegressor
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(atributos, alvo)

    # Obter a importância das features
    importancias = modelo.feature_importances_

    # Criar um DataFrame com as importâncias
    importancias_df = pd.DataFrame({
        'Atributo': atributos.columns,
        'Importancia': importancias
    })

    # Ranquear os atributos com base na importância
    importancias_df = importancias_df.sort_values(by='Importancia', ascending=False)

    # Selecionar os 'n' atributos mais importantes
    top_atributos = importancias_df.head(n)

    # Calcular a contribuição acumulada
    top_atributos['Contribuicao_Acumulada'] = top_atributos['Importancia'].cumsum()

    # Plotar o gráfico de Pareto
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Gráfico de barras
    ax1.bar(top_atributos['Atributo'], top_atributos['Importancia'], color='C0')
    ax1.set_xlabel('Atributos')
    ax1.set_ylabel('Importância', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')

    # Gráfico de linha
    ax2 = ax1.twinx()
    ax2.plot(top_atributos['Atributo'], top_atributos['Contribuicao_Acumulada'], color='C1', marker='o')
    ax2.set_ylabel('Contribuição Acumulada', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    plt.title('Gráfico de Pareto dos Atributos')
    plt.show()

# Remove os outliers da base de dados
def remover_outliers(atributos, precos):

    # Remove os outliers
    tirar_outl = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05)
    tirar_outl.fit(atributos)
    amostras_normais = tirar_outl.predict(atributos) == 1
    dados_limpos = atributos[amostras_normais]
    precos_limpos = precos[amostras_normais]

    return dados_limpos, precos_limpos

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

# Utiliza validação cruzada afim de separar a base de dados
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


# ============================
# EXECUÇÃO DO CÓDIGO
# ============================


# Trata os dados
tratar_base_dados(r"C:\Users\lucas\OneDrive - MSFT\UFSC\7º Semestre - 24.2\0 - Disciplinas\Sistemas Inteligentes\0 - Importante\Code\Regressao\base_dados.csv")

# Carregar os dados
dados = pd.read_csv(r"C:\Users\lucas\OneDrive - MSFT\UFSC\7º Semestre - 24.2\0 - Disciplinas\Sistemas Inteligentes\0 - Importante\Code\Regressao\base_dados_tratada.csv")

# Dividir os dados em treino e teste
dados_treino, dados_teste = train_test_split(dados, test_size=0.2, random_state=42)
dados_treino.to_csv("base_dados_tratada_teste.csv", index=False)
dados_teste.to_csv("base_dados_tratada_treino.csv", index=False)

# Pré-processar os dados de treino e teste
atributos_treino, precos_treino = pre_processamento(dados_treino)
atributos_teste, precos_teste = pre_processamento(dados_teste)

# Remover outliers dos dados de treino
atributos_treino_limpos, precos_treino_limpos = remover_outliers(atributos_treino, precos_treino)

# Definir os modelos a serem avaliados
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

# # Avaliar e imprimir os resultados dos modelos com a base de treino
# for nome, modelo in modelos:
#     resultados = avaliar_modelo_com_validacao_cruzada(modelo, atributos_treino_limpos, precos_treino_limpos)
#     imprimir_resultados_validacao(nome, resultados)

#     # Treinar o modelo e fazer previsões no conjunto de teste
#     modelo.fit(atributos_treino_limpos, precos_treino_limpos)

#     # Salvar o modelo treinado
#     salvar_modelo(modelo, f"{nome}_modelo.pkl")

# Carregar e testar os modelos salvos
print("-------------------------------------")
for nome, _ in modelos:
    modelo_carregado = carregar_modelo(f"{nome}_modelo.pkl")

    # Fazer previsões no conjunto de teste usando o modelo carregado
    predicoes = modelo_carregado.predict(atributos_teste)

    # Avaliar o modelo carregado com a base de teste
    resultados_teste = avaliar_modelo_com_validacao_cruzada(modelo_carregado, atributos_teste, precos_teste)
    imprimir_resultados_validacao(f"{nome} (Carregado)", resultados_teste)

    # Plotar o gráfico heatmap com o modelo carregado
    plot_heatmap_scatter(modelo_carregado, atributos_teste, precos_teste)
