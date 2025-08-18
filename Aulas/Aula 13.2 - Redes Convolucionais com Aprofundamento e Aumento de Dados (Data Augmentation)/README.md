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

```python
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
```

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
