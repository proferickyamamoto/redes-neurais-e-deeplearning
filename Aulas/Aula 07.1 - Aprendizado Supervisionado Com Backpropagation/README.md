# 🧠 Aula 07.1: Aprendizado Supervisionado com Backpropagation

## 🎯 Objetivo
Compreender o funcionamento do **algoritmo de backpropagation**, sua relação com a **função de custo** e como os pesos são ajustados em uma RNA Multicamadas (MLP) usando o **gradiente descendente**.

---

## 📚 Conteúdo Programático

### 1. Aprendizado Supervisionado
- A rede aprende com exemplos rotulados (entrada + saída esperada).
- A saída da rede é comparada com a esperada e o erro é calculado.

### 2. Função de Custo
- Mede o erro entre a saída real e a esperada.
- **Erro Quadrático Médio (MSE):**

\$$ E = \frac{1}{2} \sum (y_{esperado} - y_{obtido})^2 \$$

### 3. Gradiente Descendente
- Minimiza a função de custo atualizando os pesos:

\$$ w := w - \eta \cdot \frac{\partial E}{\partial w} \$$

- \$$(\eta)\$$ é a **taxa de aprendizado**.

### 4. Algoritmo Backpropagation
**Etapas:**
1. **Feedforward**: propaga as entradas para calcular a saída.
2. **Cálculo do erro** na saída.
3. **Backpropagation**: propaga o erro de volta pelas camadas.
4. **Atualiza os pesos** com base nos gradientes calculados.

---

## 🔧 Exemplo em Python: MLP com Backpropagation

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Dados de entrada (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Inicialização dos pesos
np.random.seed(42)
pesos0 = 2 * np.random.random((2, 3)) - 1
pesos1 = 2 * np.random.random((3, 1)) - 1
taxa_aprendizado = 0.1

# Treinamento
for i in range(10000):
    camada0 = X
    z1 = np.dot(camada0, pesos0)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, pesos1)
    a2 = sigmoid(z2)

    erro = y - a2
    delta2 = erro * sigmoid_deriv(z2)
    delta1 = delta2.dot(pesos1.T) * sigmoid_deriv(z1)

    pesos1 += a1.T.dot(delta2) * taxa_aprendizado
    pesos0 += camada0.T.dot(delta1) * taxa_aprendizado

print("Resultado final:", a2)
```

---

## 🧪 Atividade em Sala

**Título:** Treinando a RNA XOR passo a passo

### Instruções:
1. Execute o código fornecido.
2. Altere a taxa de aprendizado e observe o comportamento.
3. Plote a evolução do erro (opcional com `matplotlib`).
4. Responda:
   - A rede conseguiu aprender?
   - Com que taxa de aprendizado ela convergiu melhor?

---

## 🧠 Desafio para Casa

**Título:** Rede para classificar clima

### Objetivo:
Treinar uma RNA para classificar o clima como **ensolarado (0)** ou **chuvoso (1)** com base nas entradas:
- Temperatura (frio=0 / quente=1)
- Umidade (baixa=0 / alta=1)

### Etapas:
1. Monte uma tabela de dados com entradas binárias.
2. Treine uma RNA semelhante ao exemplo XOR.
3. Mostre o resultado final para todas as entradas.

---

## ✅ Conclusão

O algoritmo de **backpropagation** permite que as redes multicamadas aprendam a partir dos erros, ajustando os pesos de forma eficiente. Esse mecanismo é a base do aprendizado profundo e das redes neurais modernas.
