## 📘 Aulas
Os arquivos de aula estão disponíveis nesta pasta. Cada PDF contém materiais teóricos e explicações detalhadas sobre os tópicos abordados.

[Aula 1](Aula%201%20-%20Introdu%C3%A7%C3%A3o%20%C3%A0%20Intelig%C3%AAncia%20Artificial.pdf)- Abordaremos um Breafing do conteúdo de Inteligência Artificial e as tecnologias emergentes hoje em dia.

[Aula 2](Aula%202%20-%20Matem%C3%A1tica%20para%20Redes%20Neurais.pdf) - Introdução aos conceitos matemáticos fundamentais para redes neurais.

[Aula 3](Aula%2003.1%20-%20Principal%20Components%20Analysis.pdf) - 

### Aula 03.1 - Análise de Componentes Principais (PCA)
A Análise de Componentes Principais (PCA) é uma técnica estatística utilizada para reduzir a dimensionalidade de conjuntos de dados, preservando o máximo de variância possível. Isso é alcançado transformando as variáveis originais em um novo conjunto de variáveis ortogonais chamadas componentes principais.

## Implementação do PCA em Python
A seguir, apresentamos um passo a passo para implementar o PCA utilizando as bibliotecas numpy e pandas.

## 1. Importação das Bibliotecas Necessárias

import numpy as np
import pandas as pd
2. Carregamento dos Dados
Carregue seu conjunto de dados em um DataFrame do pandas. Para este exemplo, vamos criar um DataFrame fictício.

# Exemplo de dados
data = {
    'Variável_1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    'Variável_2': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
}

df = pd.DataFrame(data)
3. Padronização dos Dados
Antes de aplicar o PCA, é importante padronizar os dados para que cada variável tenha média zero e variância unitária.

# Padronização dos dados
``` python
df_padronizado = (df - df.mean()) / df.std()
```
## 4. Cálculo da Matriz de Covariância
A matriz de covariância descreve a variabilidade conjunta das variáveis no conjunto de dados.

# Cálculo da matriz de covariância
matriz_covariancia = np.cov(df_padronizado.T)
5. Cálculo dos Autovalores e Autovetores
Os autovalores e autovetores da matriz de covariância são utilizados para determinar as direções (componentes principais) e magnitudes da variância nos dados.

# Cálculo dos autovalores e autovetores
autovalores, autovetores = np.linalg.eig(matriz_covariancia)

6. Ordenação dos Autovalores e Seleção dos Componentes Principais

Ordene os autovalores em ordem decrescente e selecione os autovetores correspondentes aos maiores autovalores para formar os componentes principais.

# Criação de uma lista de tuplas (autovalor, autovetor)
pares_autovalores_autovetores = [(np.abs(autovalores[i]), autovetores[:, i]) for i in range(len(autovalores))]

# Ordenação das tuplas com base nos autovalores em ordem decrescente
pares_autovalores_autovetores.sort(key=lambda x: x[0], reverse=True)

# Seleção dos autovetores correspondentes aos maiores autovalores
autovetores_ordenados = [par[1] for par in pares_autovalores_autovetores]

7. Transformação dos Dados
Projete os dados originais no novo espaço dos componentes principais para obter as novas variáveis.

# Construção da matriz de projeção com os autovetores selecionados
matriz_projecao = np.column_stack(autovetores_ordenados)

# Transformação dos dados
dados_transformados = np.dot(df_padronizado, matriz_projecao)
8. Criação de um DataFrame com os Componentes Principais
Por fim, crie um DataFrame contendo os componentes principais resultantes.

# Criação do DataFrame com os componentes principais
```python
df_componentes_principais = pd.DataFrame(dados_transformados, columns=['Componente_1', 'Componente_2'])

```
Considerações Finais
A implementação manual do PCA fornece uma compreensão aprofundada dos passos envolvidos na redução de dimensionalidade. No entanto, em aplicações práticas, é comum utilizar bibliotecas como o scikit-learn, que oferecem implementações otimizadas e prontas para uso do PCA.

Para uma compreensão visual e prática do PCA, você pode assistir ao seguinte vídeo:

Análise de Componentes Principais (PCA) em Python - YouTube

Este vídeo oferece uma explicação detalhada e exemplos práticos que complementam os passos apresentados acima.

✍️ **Responsável:** [Erick Toshio Yamamoto]  
📅 **Última Atualização:** [25/02/2025]

