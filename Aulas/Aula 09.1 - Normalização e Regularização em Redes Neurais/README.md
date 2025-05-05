# 🧠 Aula 09.1: Normalização e Regularização em Redes Neurais

## 🎯 Objetivo
Compreender as técnicas de **normalização** e **regularização** em redes neurais, suas motivações, aplicações práticas e impacto no desempenho de modelos MLP.

---

## 📚 Conteúdo Programático

### 1. Por que normalizar os dados?
- Facilita a convergência do gradiente descendente.
- Evita que variáveis com escalas diferentes dominem o aprendizado.
- Torna o treinamento mais estável e rápido.

### Técnicas de Normalização
- **Min-Max Scaling**: escala os dados para o intervalo [0, 1].
- **Z-score (Padronização)**: transforma os dados para média 0 e desvio padrão 1.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Exemplo:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 2. Regularização
- Previne o **overfitting**, ou seja, quando a rede aprende demais os dados de treino.
- Adiciona uma penalização ao erro para evitar pesos muito grandes.

### Técnicas:
- **L1 Regularization (Lasso)**: soma dos valores absolutos dos pesos.
- **L2 Regularization (Ridge)**: soma dos quadrados dos pesos (mais comum em redes neurais).

### Exemplo com MLPClassifier:
```python
from sklearn.neural_network import MLPClassifier

modelo = MLPClassifier(hidden_layer_sizes=(10,), alpha=0.01, max_iter=1000)
# alpha é o parâmetro de regularização L2
```

---

## 💡 Impactos Práticos
- Melhora a **generalização** do modelo.
- Evita oscilações nos gradientes.
- Pode exigir ajuste fino com validação cruzada.

---

## 🧪 Atividade em Sala

**Título:** Comparando redes com e sem normalização e regularização

### Instruções:
1. Escolha qualquer dataset (como o Iris ou OpenML).
2. Treine um MLP com e sem normalização dos dados.
3. Treine com e sem o parâmetro `alpha`.
4. Compare as acurácias e o comportamento da rede (número de épocas, estabilidade).

---

## 🧠 Desafio para Casa

**Título:** Otimizando redes com regularização

### Instruções:
1. Escolha um dataset do OpenML com pelo menos 100 amostras.
2. Aplique diferentes valores de `alpha` (ex: 0.0001, 0.01, 0.1).
3. Crie gráficos comparando a acurácia de validação para cada valor.
4. Apresente suas conclusões.

---

## ✅ Conclusão

A **normalização** e **regularização** são técnicas fundamentais para tornar redes neurais mais eficientes, confiáveis e robustas. Elas evitam erros comuns e melhoram o desempenho de forma significativa.

