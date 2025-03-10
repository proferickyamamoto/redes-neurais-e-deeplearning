# 📊 Estatística Descritiva e Análise Exploratória de Dados (EDA) 🔍

## 📌 Introdução

A **Estatística Descritiva** e a **Análise Exploratória de Dados (EDA - Exploratory Data Analysis)** são etapas fundamentais na ciência de dados. Esses métodos permitem compreender a distribuição dos dados, identificar padrões, detectar outliers e tomar decisões informadas antes de aplicar algoritmos de aprendizado de máquina. 📈

---

## 📊 Estatística Descritiva

A Estatística Descritiva se divide em três categorias principais:

### 1️⃣ Medidas de Tendência Central

Essas medidas indicam o valor central de um conjunto de dados:

- **📏 Média (μ ou x̄):** Soma de todos os valores dividida pelo número total de observações.
- **📍 Mediana:** Valor central quando os dados estão ordenados.
- **🔁 Moda:** Valor que mais se repete nos dados.

### 2️⃣ Medidas de Dispersão

Essas medidas mostram a variação ou espalhamento dos dados:

- **📏 Amplitude:** Diferença entre o maior e o menor valor.
- **📊 Variância (σ²):** Média dos quadrados das diferenças entre os valores e a média.
- **📉 Desvio Padrão (σ):** Raiz quadrada da variância, indicando o quanto os dados se afastam da média.
- **📊 Coeficiente de Variação:** Desvio padrão dividido pela média, expresso em porcentagem.

### 3️⃣ Medidas de Forma

Essas medidas descrevem a distribuição dos dados:

- **📈 Assimetria (Skewness):** Indica se os dados estão inclinados para a esquerda ou direita.
- **🔄 Curtose:** Mede o "achatamento" da distribuição dos dados em relação a uma distribuição normal.

---

## 🔍 Análise Exploratória de Dados (EDA)

A Análise Exploratória de Dados é um conjunto de técnicas para visualizar e entender a estrutura dos dados antes de construir modelos preditivos.

### 📊 1. Visualizações Gráficas

- **📊 Histogramas:** Representam a distribuição dos dados.
- **📦 Boxplot (Gráfico de Caixa):** Identifica outliers e distribuição dos dados.
- **🔄 Gráficos de Dispersão (Scatter Plot):** Mostram relações entre duas variáveis.
- **📊 Pairplot:** Exibe relações entre múltiplas variáveis em um só gráfico.

### ⚠️ 2. Detecção de Outliers

- **📏 Técnica do IQR (Interquartile Range):** Considera outliers os valores fora do intervalo $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$.
- **📉 Método Z-Score:** Classifica valores com pontuação padronizada muito alta (maior que 3 ou menor que -3) como outliers.

### 🔗 3. Correlação entre Variáveis

- **📈 Coeficiente de Correlação de Pearson:** Mede a relação linear entre variáveis.
- **📊 Matriz de Correlação:** Exibe a força e direção da correlação entre variáveis.

### ❓ 4. Tratamento de Dados Faltantes

- **🚫 Remoção de registros com valores ausentes.**
- **📥 Preenchimento com a média, mediana ou moda.**
- **🧠 Interpolação ou uso de algoritmos preditivos para estimar valores ausentes.**

---

## 🖥️ Implementação em Python 🐍

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📂 Carregar os dados
df = pd.read_csv("dados.csv")

# 📊 Estatística Descritiva
descricao = df.describe()
print(descricao)

# 📈 Histograma
df.hist(figsize=(10, 6))
plt.show()

# 📦 Boxplot
sns.boxplot(data=df)
plt.show()

# 🔗 Matriz de Correlação
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## 🎯 Conclusão

A **Estatística Descritiva** e a **Análise Exploratória de Dados** são essenciais para compreender os dados antes da modelagem. Através de técnicas estatísticas e visualizações, podemos identificar padrões, outliers e relações entre variáveis, garantindo que os modelos de machine learning sejam construídos sobre uma base confiável e bem estruturada. 🚀


