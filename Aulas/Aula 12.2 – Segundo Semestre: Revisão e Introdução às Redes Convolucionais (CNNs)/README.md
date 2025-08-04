# 🧠 Aula 12.2 – Segundo Semestre: Revisão e Introdução às Redes Convolucionais (CNNs)

## 🎯 Objetivo

Retomar os principais conceitos abordados no primeiro semestre e introduzir de forma teórica e prática as Redes Neurais Convolucionais (CNNs), com base em literatura científica e aplicações reais.

---

## 📚 Parte 1 – Revisão dos Fundamentos (com exemplo de ECG e EDA com TSFEL)

Durante o primeiro semestre, exploramos a base das redes neurais artificiais (RNA), abordando estruturas feedforward, algoritmos de retropropagação (backpropagation), funções de ativação, regularização (L2, dropout, batch normalization) e otimizadores como SGD, Adam e RMSProp. A compreensão dessas estruturas é essencial para avançarmos para arquiteturas mais complexas. Como continuidade do exemplo de ECG, podemos implementar um pipeline de classificação com RNA (MLP), comparando Grid Search e Random Search, avaliando também o tempo de inferência de cada abordagem e apresentando os relatórios de desempenho.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import time

# Separar os dados
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Grid Search
param_grid = {
    'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)],
    'alpha': [0.0001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01]
}

start_grid = time.time()
grid_search = GridSearchCV(MLPClassifier(max_iter=500), param_grid, cv=3)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_grid

# Random Search
param_dist = {
    'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30), (20, 20, 20)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.0001, 0.001, 0.01]
}

start_rand = time.time()
rand_search = RandomizedSearchCV(MLPClassifier(max_iter=500), param_dist, n_iter=10, cv=3, random_state=42)
rand_search.fit(X_train, y_train)
rand_time = time.time() - start_rand

# Avaliação Grid
y_pred_grid = grid_search.predict(X_test)
print("
GridSearch Classification Report:")
print(classification_report(y_test, y_pred_grid))
print("Matriz de Confusão (GridSearch):")
print(confusion_matrix(y_test, y_pred_grid))
print(f"Tempo de Inferência GridSearch: {grid_time:.2f} segundos")

# Avaliação Random
y_pred_rand = rand_search.predict(X_test)
print("
RandomSearch Classification Report:")
print(classification_report(y_test, y_pred_rand))
print("Matriz de Confusão (RandomSearch):")
print(confusion_matrix(y_test, y_pred_rand))
print(f"Tempo de Inferência RandomSearch: {rand_time:.2f} segundos")
```

Essa análise permite comparar os métodos de busca por hiperparâmetros, observando o impacto do ajuste fino no desempenho final e no tempo computacional necessário. arquiteturas mais complexas.

Modelos como o Perceptron e MLP (Multilayer Perceptron) foram treinados usando bibliotecas como `scikit-learn`, além de implementações manuais para entender o funcionamento interno dos pesos e gradientes. Analisamos também a importância da normalização de entradas e da seleção cuidadosa de hiperparâmetros.

Utilizamos dados reais de plataformas como o OpenML e aplicamos técnicas de tuning, validação cruzada e grid search. Isso nos permitiu observar como pequenas mudanças nos hiperparâmetros afetam drasticamente o desempenho do modelo.

Como exemplo prático adicional, exploramos um sinal de ECG utilizando a biblioteca TSFEL (Time Series Feature Extraction Library) para análise exploratória de dados (EDA). Abaixo, temos um exemplo de carregamento e extração de atributos:

```python
import pandas as pd
from tsfel.feature_extraction import features
import tsfel

# Carregar o sinal ECG
from sklearn.model_selection import train_test_split
import openml

# Carregar dataset de ECG diretamente do OpenML
# Exemplo: dataset ID 44026 (ECG5000) disponível em https://www.openml.org/d/44026
dataset = openml.datasets.get_dataset(44026)
df, *_ = dataset.get_data()

# Seleciona apenas a coluna do sinal e rótulo
ecgs = df.iloc[:, :-1]  # Sinais
labels = df.iloc[:, -1]  # Rótulos

# Configurações de extração do TSFEL
cfg = tsfel.get_features_by_domain()

# Extração automática dos atributos
X = tsfel.time_series_features_extractor(cfg, ecg)
print(X.head())
```

Esse procedimento transforma um sinal bruto em um vetor de atributos interpretáveis, permitindo o uso de técnicas de aprendizado supervisionado. O próximo passo seria aplicar uma análise de componentes principais (PCA) para reduzir a dimensionalidade e visualizar a distribuição dos dados.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA dos atributos extraídos do ECG")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

Essa aplicação mostra que mesmo sinais unidimensionais como o ECG podem ser tratados com redes neurais após uma etapa eficiente de engenharia de atributos, reforçando a importância do pré-processamento antes do uso de CNNs ou MLPs.

Durante o primeiro semestre, exploramos a base das redes neurais artificiais (RNA), abordando estruturas feedforward, algoritmos de retropropagação (backpropagation), funções de ativação, regularização (L2, dropout, batch normalization) e otimizadores como SGD, Adam e RMSProp. A compreensão dessas estruturas é essencial para avançarmos para arquiteturas mais complexas.

Modelos como o Perceptron e MLP (Multilayer Perceptron) foram treinados usando bibliotecas como `scikit-learn`, além de implementações manuais para entender o funcionamento interno dos pesos e gradientes. Analisamos também a importância da normalização de entradas e da seleção cuidadosa de hiperparâmetros.

Utilizamos dados reais de plataformas como o OpenML e aplicamos técnicas de tuning, validação cruzada e grid search. Isso nos permitiu observar como pequenas mudanças nos hiperparâmetros afetam drasticamente o desempenho do modelo.

Encerramos o semestre anterior com uma introdução ao ajuste de modelos, com destaque para a busca automatizada de hiperparâmetros. Agora, com essa base, podemos nos aprofundar nas redes convolucionais, que são amplamente utilizadas para tarefas com imagens.

---

## 📘 Parte 2 – Teoria das Redes Convolucionais (CNNs)

As Redes Neurais Convolucionais (CNNs) são uma classe especializada de redes projetadas para reconhecer padrões espaciais em dados com estrutura de grade, como imagens. Elas são compostas por camadas convolucionais, camadas de pooling e camadas totalmente conectadas, que funcionam em conjunto para extrair e aprender representações hierárquicas de dados visuais.

A operação central das CNNs é a convolução, em que um filtro (kernel) é movido sobre a imagem de entrada para extrair padrões locais, como bordas, cantos e texturas. Essa abordagem permite o compartilhamento de pesos, reduzindo drasticamente a quantidade de parâmetros em relação aos MLPs tradicionais. Segundo LeCun et al. (1998), essa arquitetura foi essencial para o sucesso do modelo LeNet-5 na classificação de dígitos manuscritos \[1].

A camada de pooling (ou subamostragem) é usada para reduzir a dimensionalidade espacial das representações intermediárias. Isso não apenas diminui o custo computacional, mas também ajuda na generalização, tornando o modelo menos sensível a pequenas mudanças de posição. Dentre as técnicas de pooling, o max pooling é o mais utilizado.

[![Pooling](https://www.researchgate.net/publication/333593451/figure/fig2/AS:765890261966848@1559613876098/llustration-of-Max-Pooling-and-Average-Pooling-Figure-2-above-shows-an-example-of-max.png)](https://www.researchgate.net/publication/333593451/figure/fig2/AS:765890261966848@1559613876098/llustration-of-Max-Pooling-and-Average-Pooling-Figure-2-above-shows-an-example-of-max.png)

As CNNs tornaram-se o padrão de fato em tarefas como classificação de imagens (ImageNet), detecção de objetos (YOLO, R-CNN) e reconhecimento facial. Seu desempenho superior é sustentado por experimentos empíricos robustos, como demonstrado por Krizhevsky et al. (2012) no modelo AlexNet \[2].

![CNN nos Dígitos](https://www.louisbouchard.ai/content/images/size/w2000/2021/04/1_QPRC1lcfYxcWWPAC2hrQgg.gif)

Na prática, frameworks como TensorFlow, Keras e PyTorch facilitaram a construção e treinamento dessas redes, permitindo avanços significativos em aplicações de visão computacional em tempo real, como carros autônomos e diagnóstico médico por imagem \[3].



**Referências:**
\[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
\[2] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in neural information processing systems*.
\[3] Litjens, G. et al. (2017). A survey on deep learning in medical image analysis. *Medical image analysis*, Elsevier.

---

## 💻 Parte 3 – Implementação de uma CNN com Keras

A seguir, realizamos a construção passo a passo de uma CNN simples utilizando a biblioteca Keras, com base no dataset MNIST. A cada linha de código, explicaremos separadamente sua funcionalidade fora do bloco de código, promovendo uma melhor compreensão didática para os alunos.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```

Importamos os módulos necessários do Keras: `Sequential` para organizar o modelo em camadas sequenciais, e `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` que são as camadas típicas usadas em CNNs.

```python
model = Sequential()
```

Inicializamos o modelo sequencial. Isso significa que vamos adicionar as camadas uma após a outra, formando uma pilha linear de camadas.

```python
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
```

Adicionamos a primeira camada convolucional com 32 filtros de tamanho 3x3, ativação ReLU, e um formato de entrada de imagem 28x28 com 1 canal (tons de cinza).

```python
model.add(MaxPooling2D(pool_size=(2,2)))
```

Adicionamos uma camada de pooling com janelas de 2x2 para reduzir as dimensões espaciais pela metade, mantendo as informações mais relevantes.

```python
model.add(Flatten())
```

Transformamos a saída bidimensional das camadas anteriores em um vetor unidimensional, necessário para conectar às camadas densas.

```python
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

Adicionamos uma camada densa com 128 neurônios e ativação ReLU, seguida por uma camada de saída com 10 neurônios e ativação softmax para classificação multiclasse (0 a 9).

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Compilamos o modelo utilizando o otimizador Adam, função de perda categorical crossentropy (adequada para classificação multiclasse) e métrica de acurácia.

---

## 🧪 Atividade em Sala

**Título:** Construindo uma CNN para classificar dígitos manuscritos (MNIST)

### Instruções:

1. Carregue o dataset MNIST via `keras.datasets`.
2. Normalize os dados e aplique one-hot encoding nas saídas.
3. Use a estrutura de CNN apresentada para classificar os dígitos.
4. Avalie a acurácia no conjunto de teste.

---

## 🧠 Desafio para Casa

**Título:** Modificando a CNN com mais camadas

### Instruções:

1. Adicione uma segunda camada convolucional e de pooling ao modelo apresentado.
2. Treine novamente e compare os resultados de acurácia com o modelo original.
3. Explique as diferenças observadas em termos de generalização e overfitting.

---

## ✅ Conclusão

A introdução das CNNs marca um passo importante na disciplina, abrindo as portas para aplicações em visão computacional. A compreensão teórica e prática dessa arquitetura nos prepara para redes mais profundas e especializadas que estudaremos ao longo do semestre.
