# Aula 13.2: Redes Convolucionais com Aprofundamento e Aumento de Dados (Data Augmentation)

## 🎯 Objetivo

Explorar estratégias para melhorar a generalização de redes convolucionais (CNNs), com foco em aumento de dados, redes mais profundas e regularização. Também compreender a importância da validação cruzada e visualização de filtros aprendidos.

---

## 📘 Parte 1 – Teoria Avançada: Profundidade e Robustez das CNNs

Redes convolucionais mais profundas possibilitam a extração de representações hierárquicas mais sofisticadas. Ao empilhar múltiplas camadas convolucionais e pooling, o modelo passa a reconhecer desde bordas até formas mais abstratas. Contudo, o aumento da profundidade deve ser acompanhado de estratégias que evitem o sobreajuste, como dropout e batch normalization.

Além disso, a variação natural dos dados é um desafio importante. Se a rede só aprender imagens em condições específicas, terá dificuldade de generalizar. Para isso, técnicas de aumento de dados (data augmentation) são amplamente aplicadas. Elas introduzem variações artificiais nos dados, como rotações, zooms, translações e inversões, simulando diferentes condições de captura \[1].

Essa abordagem, como demonstrado por Simard et al. (2003), foi essencial para o bom desempenho em tarefas de escrita manual e detecção visual em condições reais \[2]. O uso de data augmentation também está presente nos modelos modernos como VGG, ResNet e Inception, onde a diversidade dos dados é fundamental para a robustez \[3].

**Referências:**
\[1] Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, Springer.
\[2] Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). Best practices for convolutional neural networks applied to visual document analysis. *ICDAR*.
\[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

---

## 💻 Parte 2 – Implementação de CNN com Aumento de Dados

````python
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Carregar e pré-processar
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Aumento de dados
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# Modelo CNN com dropout
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar com aumento de dados
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=10,
                    validation_data=(X_test, y_test),
                    verbose=1)

# Visualização da acurácia
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Acurácia com Augmentation')
plt.legend()
plt.show()

# ---

## 📡 CNN 1D para Sinais Temporais (ex: ECG)

Redes convolucionais também podem ser aplicadas em sinais temporais 1D, como o ECG. A estrutura CNN1D permite aprender padrões de forma, ritmo ou frequência em séries temporais.

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Simulando um sinal com 1000 amostras e 100 timesteps
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_classes=2, random_state=42)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 1)))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
````

---

## 🌈 Transformação de Sinal Temporal em Espectrograma para CNN 2D

Outra abordagem para trabalhar com sinais 1D em CNNs 2D é transformar o sinal em uma imagem espectral usando a transformada de Fourier ou STFT (Short-Time Fourier Transform).

```python
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

# Gerando sinal simples para exemplo
t = np.linspace(0, 1.0, 500)
signal_ex = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 80 * t)

# Gera o espectrograma
frequencies, times, Sxx = signal.spectrogram(signal_ex, fs=500)

plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))
plt.ylabel('Frequência [Hz]')
plt.xlabel('Tempo [s]')
plt.title('Espectrograma de um Sinal de Exemplo')
plt.colorbar(label='Intensidade (dB)')
plt.show()
```

A imagem gerada pode ser tratada como entrada para uma CNN convencional 2D, viabilizando a análise visual do comportamento frequencial do sinal (como em EEG, ECG, vibração, etc.).

---

## 🧪 Atividade em Sala
**Título:** Avaliando o impacto do aumento de dados

### Instruções:
1. Treine a CNN com e sem aumento de dados.
2. Compare as curvas de acurácia e perda para cada versão.
3. Escreva uma análise sobre qual modelo generaliza melhor e por quê.

---

## 🧠 Desafio para Casa
**Título:** Aumentando dados para diferentes bases

### Instruções:
1. Escolha um novo dataset de imagens (ex: Fashion-MNIST ou CIFAR-10).
2. Aplique data augmentation como no exemplo.
3. Treine uma CNN adaptada ao novo tamanho de imagem.
4. Compare com o treino sem aumento e gere gráficos e análise crítica.

---

## ✅ Conclusão

O uso de aumento de dados se mostrou uma estratégia poderosa para ampliar a robustez das CNNs, especialmente quando os dados disponíveis são limitados. Além disso, redes mais profundas exigem regularizações como dropout e validação cuidadosa para manter a capacidade de generalização.


