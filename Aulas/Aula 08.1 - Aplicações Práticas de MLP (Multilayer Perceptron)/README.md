# 🧠 Aula 08.1: Aplicações Práticas de MLP (Multilayer Perceptron)

## 🎯 Objetivo
Explorar **aplicações reais** das redes neurais multicamadas (MLPs), reforçando os conceitos de aprendizado supervisionado e demonstrando como essas arquiteturas podem resolver problemas complexos de classificação e regressão.

---

## 📚 Conteúdo Programático

### 1. O que é uma MLP?
- Rede neural com pelo menos **uma camada oculta**.
- Cada camada é composta por neurônios com funções de ativação (ex: ReLU, sigmoide).
- Aprendizado via **backpropagation**.

### 2. Exemplos de Aplicação
- Classificação de padrões (ex: letras manuscritas, reconhecimento facial)
- Diagnóstico médico
- Previsão de séries temporais (ex: temperatura, vendas)
- Reconhecimento de voz

---

## 🏗️ Estrutura Típica de uma MLP

```
Entrada → Camada Oculta (n neurônios) → Camada de Saída
```

- Função de ativação recomendada para a camada oculta: `ReLU`
- Função para a saída:
  - **Funções de ativação recomendadas para a saída:**

- `Sigmoid`: usada em **classificações binárias**. Produz uma saída entre 0 e 1, interpretável como probabilidade.
  ![Sigmoid](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

- `Softmax`: usada em **classificações multiclasse**. Normaliza as saídas em um vetor de probabilidades que somam 1.
  ![Softmax example](https://upload.wikimedia.org/wikipedia/commons/8/8a/Softmax_function.svg)

- `Linear`: usada em **regressão**, ou seja, problemas com saída numérica contínua. A saída é diretamente proporcional à soma ponderada das ativações.
  ![Linear function](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_function.svg)

---

## 💻 Exemplo Prático: Classificação de Flores (Iris Dataset)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Carregar dados
iris = load_iris()
X = iris.data
y = iris.target

# Pré-processamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Criar modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Avaliação
y_pred = mlp.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
```

### Resultado esperado:
> Acurácia acima de 90% para classificação das espécies de flores.

---

## 🧪 Atividade em Sala

**Título:** Classificando pacientes com MLP

### Instruções:
1. Criar um dataset com variáveis como: febre, tosse, dor de cabeça (0 ou 1).
2. Rotular os dados como “doente” (1) ou “saudável” (0).
3. Utilizar `MLPClassifier` para treinar um modelo.
4. Avaliar a acurácia do modelo com dados de teste.

**Desafio extra:** Testar diferentes quantidades de neurônios na camada oculta e comparar os resultados.

---

## 🧠 Desafio para Casa

**Título:** Previsão de Vendas com MLPRegressor

### Objetivo:
Usar uma rede neural para prever o valor de vendas futuras com base em entradas como mês, promoções, clima.

### Instruções:
1. Criar um pequeno dataset fictício com entradas contínuas.
2. Utilizar `MLPRegressor` do `sklearn.neural_network`.
3. Avaliar o erro com `mean_squared_error`.

---

## ✅ Conclusão

Redes do tipo **MLP** são extremamente versáteis e aplicáveis em diversas áreas. Ao entender como ajustar hiperparâmetros e interpretar os resultados, os alunos podem aplicar essas redes com confiança em problemas reais.

