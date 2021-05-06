# -*- coding: utf-8 -*-

"""
Created on Sun May  2 20:30:52 2021

@author: T-Gamer
"""


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score


# Importando dados

dados = pd.read_csv("conjunto_de_treinamento.csv")
dados_teste = pd.read_csv("conjunto_de_teste.csv")


# Limpando dados
dados.set_index('id_solicitante', inplace=True)
dados = dados.drop(['grau_instrucao',
                    'possui_telefone_celular',
                    'qtde_contas_bancarias_especiais'],axis=1)


dados.loc[dados['sexo'] == ' ','sexo'] = 'N'
dados.loc[dados['idade'] < 17,'idade'] = 17
dados.loc[dados['qtde_dependentes'] > 14,'qtde_dependentes'] = 14

# Analise de dados

for coluna in dados.columns:
    print('Valores de ' + coluna)
    print(dados[coluna].value_counts())
    


# One Hot Encoding

dados = pd.get_dummies(dados,columns=['produto_solicitado',
                                      'dia_vencimento',
                                      'forma_envio_solicitacao',
                                      ''])

# Binarizar

binalizer = LabelBinarizer()

for coluna in ['intl_plan','voice_mail_plan']:
    dados[coluna] = binalizer.fit_transform(dados[coluna])
    
    
# Verificando diferenças nas médias
'''
for coluna in dados.columns:
    if coluna != 'inadimplente':
        print("Diferença de " + coluna)
        print(abs(dados.groupby('inadimplente').mean()[coluna][0] - dados.groupby('inadimplente').mean()[coluna][1])*(100/dados.groupby('inadimplente').mean()[coluna][1]))
        print('\n')

'''
