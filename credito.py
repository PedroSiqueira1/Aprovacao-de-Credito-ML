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
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Importando dados

dados = pd.read_csv("conjunto_de_treinamento.csv")
dados_teste = pd.read_csv("conjunto_de_teste.csv")
exemplo = pd.read_csv("exemplo_arquivo_respostas.csv")


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
dados['grau_instrucao_companheiro'] = dados['grau_instrucao_companheiro'].fillna(0)
dados['ocupacao'] = dados['ocupacao'].fillna(2.0)  
dados['meses_na_residencia'] = dados['meses_na_residencia'].fillna(10.0)  

dados_teste.set_index('id_solicitante', inplace=True)
dados_teste = dados_teste.drop(['grau_instrucao',   # Falha na extração
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
                    'local_onde_trabalha' # One Hot encoding muito grande
                    ],axis=1) 


dados_teste.loc[dados_teste['sexo'] == ' ','sexo'] = 'N'
dados_teste.loc[dados_teste['idade'] < 17,'idade'] = 17
dados_teste.loc[dados_teste['qtde_dependentes'] > 14,'qtde_dependentes'] = 14
dados_teste.loc[dados_teste['estado_onde_nasceu'] == ' ','estado_onde_nasceu'] = 'N'
dados_teste['grau_instrucao_companheiro'] = dados_teste['grau_instrucao_companheiro'].fillna(0)
dados_teste['ocupacao'] = dados_teste['ocupacao'].fillna(2.0)  
dados_teste['meses_na_residencia'] = dados_teste['meses_na_residencia'].fillna(10.0)  

'''
# Analise de dados

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

dados_teste = pd.get_dummies(dados_teste,columns=['produto_solicitado',
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
    dados_teste[coluna] = binalizer.fit_transform(dados_teste[coluna])
    
    

# Verificando diferenças nas médias


for coluna in dados.columns:
    if coluna != 'inadimplente' and dados[coluna].dtype != 'O':
        print("Diferença de " + coluna)
        print(abs(dados.groupby('inadimplente').mean()[coluna][0] - dados.groupby('inadimplente').mean()[coluna][1])*(100/dados.groupby('inadimplente').mean()[coluna][1]))
        print('\n')

# Atributos selecionados

atributos_selecionados = [ 'dia_vencimento',
                           'idade',
                           'qtde_dependentes',
                          # 'estado_onde_nasceu',
                          # 'estado_onde_reside',
                           'possui_telefone_residencial',
                           'meses_na_residencia',
                           'possui_email',
                           'renda_mensal_regular',
                           'renda_extra',
                           'possui_cartao_visa',
                           'possui_cartao_mastercard',
                           'possui_cartao_diners',
                           'possui_cartao_amex',
                           'possui_outros_cartoes',
                           'qtde_contas_bancarias',
                           'valor_patrimonio_pessoal',
                           'possui_carro',
                           'vinculo_formal_com_empresa',
                           'possui_telefone_trabalho',
                           'grau_instrucao_companheiro',
                           'produto_solicitado_1',
                           'produto_solicitado_2',
                           'produto_solicitado_7',
                           'forma_envio_solicitacao_correio',
                           'forma_envio_solicitacao_internet',
                           'forma_envio_solicitacao_presencial',
                           'sexo_F',
                           'sexo_M',
                           'sexo_N',
                           'estado_civil_0',
                           'estado_civil_1',
                           'estado_civil_2',
                           'estado_civil_3',
                           'estado_civil_4',
                           'estado_civil_5',
                           'estado_civil_6',
                           'estado_civil_7',
                           'nacionalidade_0',
                           'nacionalidade_1',
                           'nacionalidade_2',
                           'tipo_residencia_0.0',
                           'tipo_residencia_1.0',
                           'tipo_residencia_2.0',
                           'tipo_residencia_3.0',
                           'tipo_residencia_4.0',
                           'tipo_residencia_5.0',
                           'ocupacao_0.0',
                           'ocupacao_1.0',
                           'ocupacao_2.0',
                           'ocupacao_3.0',
                           'ocupacao_4.0',
                           'ocupacao_5.0',
                           'inadimplente']


dados = dados[atributos_selecionados]
dados_teste = dados_teste[atributos_selecionados[0:-1]]

# Separando alvo e atributos

X = dados.iloc[:,dados.columns != 'inadimplente'].values
y = dados.iloc[:,dados.columns == 'inadimplente'].values.ravel()

# Separando treino e teste

X_train, X_test, y_train,y_test = train_test_split(X,y, train_size=15000, random_state = 42)


# Ajustar escala

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''

'''
# Validação cruzada
'''
for k in range(550,600):
    
    classificador = KNeighborsClassifier(
        n_neighbors = k)
    
    classificador = classificador.fit(X_train,y_train)
    scores = cross_val_score(classificador,X,y, cv=5)
    print('k =%2d' % k,
          'scores =', scores,
          'acuracia média = %6.1f' % (100*sum(scores)/5)
          )
'''
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)


# KNN


classificadorKNN = KNeighborsClassifier(
        n_neighbors = 550)

classificadorKNN = classificadorKNN.fit(X,y)
resposta_knn = classificadorKNN.predict(dados_teste)



# Grid SVM
'''
parameters = {
    "kernel": ["rbf"],
    "C": [1,10,100,1000],
    "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }

grid = GridSearchCV(SVC(), parameters, cv=5, verbose=2)
grid.fit(X, y)
'''
# Melhores Parametros {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}

# SVM Kernel

classificadorSVM = SVC(C=100, gamma=0.001)
classificadorSVM.fit(X,y)
resposta_svm = classificadorSVM.predict(dados_teste)

# Rnadom Forest

classificadorRF = RandomForestClassifier()
classificadorRF.fit(X,y)
respostaRF = classificadorRF.predict(dados_teste)

# Confusion Matrix

# Resposta para csv
resposta_final = respostaRF

exemplo['inadimplente'] = resposta_final
exemplo.to_csv('resposta1.csv', index= False)