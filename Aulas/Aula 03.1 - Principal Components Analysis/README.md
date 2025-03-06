# 📌 Análise de Componentes Principais (PCA)

## 🎯 Objetivos da Aula
Nesta aula, vamos abordar os seguintes tópicos:
- O conceito de **redução de dimensionalidade** e sua importância em Machine Learning.
- A **Análise de Componentes Principais (PCA)** como método estatístico para simplificar conjuntos de dados.
- Fundamentos matemáticos: **variância, covariância, autovalores e autovetores**.
- Implementação prática do **PCA em Python**.

---

## 🔍 1. O que é PCA e por que usá-lo?
A **Análise de Componentes Principais (PCA)** é uma técnica estatística usada para **reduzir a dimensionalidade dos dados**, mantendo a maior quantidade possível de variância.

### ✅ **Motivações para o uso do PCA:**
1. **Redução da dimensionalidade** → Facilita a visualização e análise de dados.
2. **Eliminação de multicolinearidade** → Remove redundâncias entre variáveis correlacionadas.
3. **Melhoria na performance de modelos** → Reduz o risco de sobreajuste.
4. **Visualização de dados complexos** → Permite projetar dados multidimensionais em um espaço menor.

---

## 📊 2. Fundamentos Matemáticos do PCA
### 📌 **Variância**
A **variância** mede a dispersão dos dados em relação à média:

```math
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
```

Onde:
- $$\ x_i \$$ são os valores individuais da variável.
- $$\ \bar{x} \$$ é a média dos valores.

### 📌 **Covariância**
A **covariância** mede como duas variáveis variam juntas:

```math
	\text{cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
```

Uma **matriz de covariância** captura a relação entre múltiplas variáveis:

```math
C = 
\begin{bmatrix}
	\text{Var}(X_1) 	& 	\text{Cov}(X_1, X_2) 	& \dots \\
	\text{Cov}(X_2, X_1) 	& 	\text{Var}(X_2) 	& \dots \\
	\dots 			& 	\dots 			& \ddots
\end{bmatrix}
```

### 📌 **Autovalores e Autovetores**
Os **autovalores** ($$\\lambda \$$) e **autovetores** ($$\ v \$$) de uma matriz $$\ A \$$ satisfazem:
```math
A v = \lambda v
```

No contexto do PCA:
- **Autovetores** representam as **direções principais** dos dados.
- **Autovalores** indicam **quanta variância** cada direção retém.

---

## 🔬 3. Passos para Implementar o PCA
O PCA é executado seguindo os passos:

1️⃣ **Padronizar os dados** (média 0 e variância 1).  
2️⃣ **Calcular a matriz de covariância** entre as variáveis.  
3️⃣ **Encontrar os autovalores e autovetores** da matriz de covariância.  
4️⃣ **Ordenar os autovalores em ordem decrescente** e selecionar os principais.  
5️⃣ **Projetar os dados no novo espaço** formado pelos componentes principais.

---

## 🛠 4. Implementação Prática do PCA em Python
Aqui está um exemplo de como implementar o PCA manualmente:

```python
import numpy as np
import pandas as pd

# Exemplo de dados
data = {
    'Variável_1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    'Variável_2': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
}

df = pd.DataFrame(data)

# Padronização dos dados
df_padronizado = (df - df.mean()) / df.std()

# Cálculo da matriz de covariância
matriz_covariancia = np.cov(df_padronizado.T)

# Cálculo dos autovalores e autovetores
autovalores, autovetores = np.linalg.eig(matriz_covariancia)

# Ordenação dos autovalores e autovetores
pares_autovalores = [(autovalores[i], autovetores[:, i]) for i in range(len(autovalores))]
pares_autovalores.sort(key=lambda x: x[0], reverse=True)

# Transformação dos dados
matriz_projecao = np.column_stack([pares_autovalores[i][1] for i in range(len(autovalores))])
dados_transformados = np.dot(df_padronizado, matriz_projecao)

# Criar DataFrame com os componentes principais
df_pca = pd.DataFrame(dados_transformados, columns=['Componente_1', 'Componente_2'])
print(df_pca)
```

📌 **Notebook:** [notebooks/principal_components_analysis.ipynb](notebooks/principal_components_analysis.ipynb)

---

## 📚 5. Recursos Recomendados
📖 **Livros:**
- “Mathematics for Machine Learning” – Marc Peter Deisenroth.
- “Deep Learning” – Ian Goodfellow.

🎥 **Cursos Online:**
- [Coursera - Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning)
- [FastAI - Introduction to ML](https://course.fast.ai/)

---

## 🚀 6. Próximos Passos
Na próxima aula, exploraremos como integrar o **sklearn** ao PCA e utilizar a técnica para **pré-processamento de dados** antes de treinar redes neurais!

---
📝 Autor: **Prof. Erick Toshio Yamamoto**
📅 Data: 06/03/2025
📌 Disciplina: Redes Neurais e Deep Learning
