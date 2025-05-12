# 🧠 Última Aula 11.1: Otimizadores Avançados e Ajuste de Hiperparâmetros em Redes Neurais

## 🎯 Objetivo

Encerrar o semestre com uma aula dedicada ao refinamento e melhoria de redes neurais, abordando **otimizadores modernos**, técnicas de **tuning de hiperparâmetros** e ferramentas automatizadas para busca de melhores modelos.

---

## 📚 Conteúdo Programático

### 1. Otimizadores Avançados

| Otimizador   | Característica Principal                                                 |
| ------------ | ------------------------------------------------------------------------ |
| **SGD**      | Gradiente simples com taxa fixa                                          |
| **Momentum** | Adiciona "inércia" à atualização                                         |
| **RMSProp**  | Ajusta a taxa de aprendizado com base na média quadrática dos gradientes |
| **Adam**     | Combina Momentum + RMSProp; muito usado em deep learning                 |

### Exemplo com Keras:

```python
from keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
```

---

### 2. Hiperparâmetros em Redes Neurais

* Número de neurônios nas camadas
* Número de camadas ocultas
* Taxa de aprendizado (learning rate)
* Funções de ativação
* Tamanho do batch e número de épocas

### Técnicas de Ajuste

* **Grid Search**
* **Random Search**
* **Bayesian Optimization** (ex: Optuna, Hyperopt)

---

## 🔧 Ferramentas de Tuning Automatizado

### Exemplo com GridSearchCV (sklearn):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'alpha': [0.0001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

grid = GridSearchCV(MLPClassifier(max_iter=500), param_grid, cv=3)
grid.fit(X_train, y_train)
print("Melhores parâmetros:", grid.best_params_)
```

---

## 🧪 Atividade em Sala

**Título:** Encontrando a melhor rede

### Instruções:

1. Escolha um dataset conhecido (OpenML, sklearn).
2. Teste 3 combinações diferentes de hiperparâmetros (camadas, taxa de aprendizado, etc.).
3. Avalie o desempenho com acurácia e matriz de confusão.
4. Discuta o impacto dos hiperparâmetros escolhidos.

---

## 🧠 Desafio Final para Casa

**Título:** Competição Final – Ajuste de Modelo Neural

### Instruções:

1. Cada grupo escolherá um dataset de livre escolha.
2. Implementar uma rede neural com pelo menos 2 camadas ocultas.
3. Ajustar hiperparâmetros manualmente ou com busca automatizada.
4. Comparar pelo menos 2 otimizadores diferentes.
5. Gerar um relatório e gráfico comparando desempenho.

**Avaliação final:** modelo mais bem ajustado + clareza na apresentação dos resultados.

---

## ✅ Conclusão

A aula final reúne as técnicas mais modernas de otimização e busca de hiperparâmetros, essenciais para alcançar redes neurais realmente eficazes em aplicações práticas. Encerra-se assim o ciclo de fundamentos, prática e refino de modelos em aprendizado profundo.
