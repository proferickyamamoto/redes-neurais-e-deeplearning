
# Aula 03.1- Análise de Componentes Principais (PCA) 

A Análise de Componentes Principais (PCA) é uma técnica estatística utilizada para reduzir a dimensionalidade de conjuntos de dados, preservando o máximo de variância possível. Isso é alcançado transformando as variáveis originais em um novo conjunto de variáveis ortogonais chamadas componentes principais.

## Implementação do PCA em Python

A seguir, apresentamos um passo a passo para implementar o PCA utilizando as bibliotecas numpy e pandas.

### 1. Importação das Bibliotecas Necessárias

```python
import numpy as np
import pandas as pd
```
### 2. Carregamento dos Dados

Carregue seu conjunto de dados em um DataFrame do pandas. Para este exemplo, vamos criar um DataFrame fictício.
```python
# Exemplo de dados
data = {
    'Variável_1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    'Variável_2': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
}

df = pd.DataFrame(data)
```
### 3. Padronização dos Dados

Antes de aplicar o PCA, é importante padronizar os dados para que cada variável tenha média zero e variância unitária.
``` python
# Padronização dos dados
df_padronizado = (df - df.mean()) / df.std()
```
### 4. Cálculo da Matriz de Covariância

A matriz de covariância descreve a variabilidade conjunta das variáveis no conjunto de dados.
``` python
# Cálculo da matriz de covariância
matriz_covariancia = np.cov(df_padronizado.T)
```
### 5. Cálculo dos Autovalores e Autovetores

Os autovalores e autovetores da matriz de covariância são utilizados para determinar as direções (componentes principais) e magnitudes da variância nos dados.
``` python
# Cálculo dos autovalores e autovetores
autovalores, autovetores = np.linalg.eig(matriz_covariancia)
``` 
### 6. Ordenação dos Autovalores e Seleção dos Componentes Principais

Ordene os autovalores em ordem decrescente e selecione os autovetores correspondentes aos maiores autovalores para formar os componentes principais.
``` python
# Criação de uma lista de tuplas (autovalor, autovetor)
pares_autovalores_autovetores = [(np.abs(autovalores[i]), autovetores[:, i]) for i in range(len(autovalores))]
# Ordenação das tuplas com base nos autovalores em ordem decrescente
pares_autovalores_autovetores.sort(key=lambda x: x[0], reverse=True)
# Seleção dos autovetores correspondentes aos maiores autovalores
autovetores_ordenados = [par[1] for par in pares_autovalores_autovetores]
```
### 7. Transformação dos Dados

Projete os dados originais no novo espaço dos componentes principais para obter as novas variáveis.
``` python
# Construção da matriz de projeção com os autovetores selecionados
matriz_projecao = np.column_stack(autovetores_ordenados)
# Transformação dos dados
dados_transformados = np.dot(df_padronizado, matriz_projecao)
```
### 8. Criação de um DataFrame com os Componentes Principais

Por fim, crie um DataFrame contendo os componentes principais resultantes.
```python
# Criação do DataFrame com os componentes principais
df_componentes_principais = pd.DataFrame(dados_transformados, columns=['Componente_1', 'Componente_2'])
```
# Considerações Finais
A implementação manual do PCA fornece uma compreensão aprofundada dos passos envolvidos na redução de dimensionalidade. No entanto, em aplicações práticas, é comum utilizar bibliotecas como o scikit-learn, que oferecem implementações otimizadas e prontas para uso do PCA.


