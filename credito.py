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
dados = dados.drop(['grau_instrucao',   # Falha na extração
                    'possui_telefone_celular',  # Falha na extração
                    'qtde_contas_bancarias_especiais', # Idêntico a coluna 'qntde_contas_bancarias'
                    'tipo_endereco',  # Vasta maioria com o atributo 1
                    'codigo_area_telefone_residencial', # One hot encoding ficaria muito grande
                    'estado_onde_trabalha', # Muitos valores vazios
                    'codigo_area_telefone_trabalho', # Muitos valores vazios
                    'meses_no_trabalho', # Vasta maioria com valor 0
                    'profissao', # One Hot encoding muito grande
                    'profissao_companheiro', # One Hot encoding muito grande
                    'local_onde_reside', # One Hot encoding muito grande
                    'local_onde_trabalha'],axis=1) # One Hot encoding muito grande


dados.loc[dados['sexo'] == ' ','sexo'] = 'N'
dados.loc[dados['idade'] < 17,'idade'] = 17
dados.loc[dados['qtde_dependentes'] > 14,'qtde_dependentes'] = 14
dados.loc[dados['estado_onde_nasceu'] == ' ','estado_onde_nasceu'] = 'N'


# Analise de dados
'''
for coluna in dados.columns:
    print('Valores de ' + coluna)
    print(dados[coluna].value_counts())
    
'''


# One Hot Encoding

dados = pd.get_dummies(dados,columns=['produto_solicitado',
                                      'forma_envio_solicitacao',
                                      'sexo',
                                      'estado_civil',
                                      'nacionalidade',
                                      'tipo_residencia',
                                      'ocupacao'])

# Binarizarda
binalizer = LabelBinarizer()

for coluna in ['possui_telefone_residencial',
               'vinculo_formal_com_empresa',
               'possui_telefone_trabalho']:
    dados[coluna] = binalizer.fit_transform(dados[coluna])
    
'''  
# Verificando diferenças nas médias



for coluna in dados.columns:
    if coluna != 'inadimplente' and dados[coluna].dtype != 'O':
        print("Diferença de " + coluna)
        print(abs(dados.groupby('inadimplente').mean()[coluna][0] - dados.groupby('inadimplente').mean()[coluna][1])*(100/dados.groupby('inadimplente').mean()[coluna][1]))
        print('\n')

'''