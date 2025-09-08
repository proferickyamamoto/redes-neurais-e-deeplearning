# Aula 14.2 – Redes Neurais Recorrentes (RNNs) e LSTM

## 🎯 Objetivo

Compreender o funcionamento das Redes Neurais Recorrentes (RNNs) e suas variantes mais utilizadas, como LSTM (Long Short-Term Memory), para modelagem de dados sequenciais e temporais, como séries de tempo, linguagem natural e sinais biomédicos.

---

## 📘 Parte 1 – Teoria das RNNs e LSTMs

As Redes Neurais Recorrentes (RNNs) foram projetadas para lidar com dados sequenciais, onde a ordem dos elementos é relevante. Diferente das redes feedforward, as RNNs possuem conexões recorrentes que permitem manter uma **memória do estado anterior**, possibilitando o aprendizado de dependências temporais. Isso as torna adequadas para tarefas como previsão de séries temporais, reconhecimento de fala e processamento de linguagem natural \[1].

Entretanto, as RNNs tradicionais enfrentam problemas de **desvanecimento e explosão do gradiente**, dificultando o aprendizado de dependências de longo prazo. Para resolver essas limitações, foram propostas variantes como **LSTM (Long Short-Term Memory)** \[2] e **GRU (Gated Recurrent Unit)** \[3].

<img width="1400" height="544" alt="image" src="https://github.com/user-attachments/assets/3f374388-a519-4089-960a-f1c71e259e78" />


As LSTMs utilizam um conjunto de portas (input, forget e output gates) que regulam o fluxo de informação e permitem o aprendizado de dependências longas de forma estável. Esse mecanismo tornou as LSTMs o padrão de fato em muitas aplicações envolvendo sequências, incluindo tradução automática, análise de sentimentos e previsão de sinais biomédicos, como ECGs e EEGs.

**Referências:**
\[1] Elman, J. L. (1990). Finding structure in time. *Cognitive Science*.
\[2] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*.
\[3] Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP*.

---

## 💻 Parte 2 – Implementação de uma RNN simples e uma LSTM

### Exemplo com sequência numérica simples

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

# Criando dados sequenciais artificiais
X = np.array([[[i+j] for i in range(5)] for j in range(100)])  # 100 amostras de sequências de tamanho 5
y = np.array([j+5 for j in range(100)])  # próximo número da sequência

# Normalização
X = X / 100.0
y = y / 100.0

# RNN simples
model_rnn = Sequential([
    SimpleRNN(10, activation='tanh', input_shape=(5,1)),
    Dense(1)
])
model_rnn.compile(optimizer='adam', loss='mse')
model_rnn.fit(X, y, epochs=50, verbose=0)

# LSTM
model_lstm = Sequential([
    LSTM(10, activation='tanh', input_shape=(5,1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=50, verbose=0)

print("Predição RNN:", model_rnn.predict(X[:1]))
print("Predição LSTM:", model_lstm.predict(X[:1]))
```

Explicação: o código acima cria sequências de números inteiros e treina uma RNN simples e uma LSTM para prever o próximo número. Ambas conseguem aprender o padrão, mas a LSTM tende a apresentar maior estabilidade em sequências mais longas.

---

## 🧪 Atividade em Sala

**Título:** Previsão de série temporal com LSTM

### Instruções:

1. Gere uma série temporal simples, como uma onda seno com ruído.
2. Prepare os dados em janelas de tempo (ex: 20 passos passados para prever o próximo).
3. Treine um modelo LSTM para prever os próximos valores.
4. Compare a previsão com os valores reais utilizando gráficos.

---

## 🧠 Desafio para Casa

**Título:** Classificação de Sentimentos com LSTM

### Instruções:

1. Utilize o dataset IMDb disponível no Keras (`keras.datasets.imdb`).
2. Pré-processe os dados (padding das sequências).
3. Treine uma LSTM para classificar críticas como positivas ou negativas.
4. Avalie a acurácia no conjunto de teste.

---

## ✅ Conclusão

As RNNs e suas variantes, como as LSTMs, revolucionaram a modelagem de dados sequenciais ao possibilitar o aprendizado de dependências temporais de curto e longo prazo. Sua aplicação é ampla, cobrindo desde a previsão de séries temporais até o processamento de linguagem natural, e estabelecendo as bases para arquiteturas mais avançadas que veremos adiante, como Transformers.
