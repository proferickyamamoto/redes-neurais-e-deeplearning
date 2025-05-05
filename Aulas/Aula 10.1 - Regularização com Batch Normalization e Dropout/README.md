# 🧠 Aula 10.1: Regularização com Batch Normalization e Dropout

## 🎯 Objetivo
Aprofundar os conhecimentos em regularização, apresentando as técnicas de **Batch Normalization** e **Dropout**, que ajudam a melhorar a estabilidade e a capacidade de generalização de redes neurais.

---

## 📚 Conteúdo Programático

### 1. Batch Normalization
- Normaliza as ativações de cada camada durante o treinamento.
- Ajuda a estabilizar o aprendizado e acelerar a convergência.
- Aplicada entre a multiplicação de pesos e a função de ativação.

### Fórmula:
\$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\$$
- \$$\mu_B \$$: média do mini-batch
- \$$\sigma_B^2 \$$: variância do mini-batch

### Exemplo com Keras:
```python
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],)))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
```

---

### 2. Dropout
- Desativa aleatoriamente neurônios durante o treinamento.
- Ajuda a evitar coadaptação entre neurônios.
- Estimula a rede a aprender representações mais robustas.

### Exemplo com Keras:
```python
from keras.layers import Dropout

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

---

## ⚠️ Considerações
- **Batch Normalization** e **Dropout** não devem ser usados indiscriminadamente.
- Em alguns casos, podem competir entre si ou ser redundantes.
- Testar e validar diferentes combinações é essencial.

---

## 🧪 Atividade em Sala

**Título:** Testando as regularizações modernas

### Instruções:
1. Escolha um dataset simples (ex: OpenML, Iris ou Breast Cancer).
2. Implemente um modelo com:
   - Apenas Batch Normalization
   - Apenas Dropout
   - Ambos combinados
3. Compare o desempenho de cada versão.
4. Justifique qual combinação apresentou melhor generalização.

---

## 🧠 Desafio para Casa

**Título:** Rede neural robusta para classificação

### Instruções:
1. Escolha um dataset com pelo menos 3 classes e 200 amostras.
2. Implemente um modelo com camadas ocultas, Batch Normalization e Dropout.
3. Use validação cruzada para avaliar a robustez.
4. Crie gráficos de comparação.

---

## ✅ Conclusão

Batch Normalization e Dropout são técnicas modernas de regularização que, se bem aplicadas, tornam redes neurais mais estáveis, eficientes e capazes de generalizar melhor em problemas reais.

