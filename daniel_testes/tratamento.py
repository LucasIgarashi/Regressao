import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

minha_base = pd.read_csv("/home/daniel-porto/Sistemas_inteligentes/trab_tratamento/train.csv")

#Separando Rotulos e Dados
data = np.array[minha_base.iloc[1:-1:]]
labels = np.array(minha_base.iloc[1:, ])