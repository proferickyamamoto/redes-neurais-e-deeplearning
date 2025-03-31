# 🧠 Introdução às Redes Neurais Artificiais (RNAs)

As Redes Neurais Artificiais são modelos computacionais inspirados na estrutura do cérebro humano. Elas são compostas por unidades chamadas **neurônios artificiais**, que estão interconectadas por **pesos sinápticos**. Esses modelos são capazes de **aprender padrões a partir de dados** e fazer previsões, classificações ou regressões.

---

## 🧬 Inspiração Biológica

| Neurônio Biológico     | Neurônio Artificial           |
|-------------------------|-------------------------------|
| Dendritos: recebem sinais | Entradas (features do dado)    |
| Corpo celular: processa   | Soma ponderada das entradas   |
| Axônio: envia sinal       | Saída do neurônio             |

---

## ⚙️ Arquitetura de uma RNA

Uma rede neural é composta por camadas:

- **Camada de entrada**: recebe os dados.
- **Camadas ocultas**: fazem transformações nos dados.
- **Camada de saída**: gera o resultado final.

Cada conexão entre neurônios possui um **peso**, que é ajustado durante o treinamento.

### 🧮 Modelo Matemático de um Neurônio

\$$ u = \sum_{i=1}^{n} x_i w_i \$$

\$$ y = f(u + b) \$$

Onde:
- \$$( x_i )\$$: entrada
- \$$( w_i )\$$: peso associado à entrada
- \$$( b )\$$: viés (bias)
- \$$( f )\$$: função de ativação (ex.: sigmoide, ReLU)

---

## 🔧 Funções de Ativação

- **Step Function (limiar):** binária, simples, usada no Perceptron original.
- **Sigmoide:** \$$[ f(x) = \frac{1}{1 + e^{-x}} ]\$$
- **ReLU (Rectified Linear Unit):** \$$[ f(x) = \max(0, x) ]\$$

---

## 🔄 Etapa de Feedforward

A propagação direta (feedforward) consiste em:
1. Receber as entradas na camada de entrada.
2. Multiplicar pelas conexões com os pesos.
3. Somar os resultados.
4. Aplicar a função de ativação.
5. Enviar a saída para a próxima camada.

Essa etapa ocorre **da entrada até a saída final**, sem ajuste de pesos (sem aprendizado).

---

## 💾 Redes Neurais com Memória RAM (RAM-based Neural Networks)

As **Redes Neurais RAM** são arquiteturas alternativas às redes convencionais baseadas em pesos. Inspiradas nos conceitos de **memória de acesso aleatório (RAM)**, essas redes armazenam padrões diretamente na memória e os acessam por endereçamento.

### Características:
- Não utilizam pesos; os neurônios são implementados com células de memória.
- A entrada é usada como endereço para recuperar o conteúdo armazenado.
- Aprendizado simples: armazenar a presença de padrões na memória.

### Exemplos:
- **WiSARD**: Rede baseada em RAM com neurônios que armazenam padrões binários.
- **PLN (Probabilistic Logic Networks)**: versão probabilística que lida com incertezas.

### Vantagens:
- Alta velocidade de aprendizado (one-shot learning).
- Simplicidade de implementação em hardware.

### Limitações:
- Pouca generalização em entradas não vistas.
- Pode exigir grande quantidade de memória em problemas complexos.

As RNAs RAM são eficazes em aplicações de reconhecimento de padrões binários, como reconhecimento de caracteres, biometria e detecção de padrões visuais simples.

---

## 💻 Implementação em Python (Feedforward)

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.where(x >= 0, 1, 0)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.activation = activation

    def activate(self, x):
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'step':
            return step_function(x)
        else:
            raise ValueError("Função de ativação desconhecida")

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activate(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.activate(self.z2)
        return self.output

# Exemplo de uso com função degrau:
X = np.array([[0, 1]])  # entrada binária
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, activation='step')
saida = nn.feedforward(X)
print("Saída da rede:", saida)
```

---

## 🧪 Atividades em Sala

### ✔️ Atividade 1 – Entendendo o Perceptron
- Construa, com papel e caneta, o modelo de um perceptron com 2 entradas.
- Atribua valores fictícios de entrada e pesos.
- Calcule manualmente a saída usando a função degrau.

### ✔️ Atividade 2 – Teste da Rede em Python
- Execute o código da rede neural em Python (fornecido acima).
- Altere as entradas e analise como a saída da rede muda.
- Documente os testes em uma tabela com entrada → saída.

### ✔️ Atividade 3 – Visualização da Arquitetura
- Em grupos, desenhem a arquitetura da rede neural do exemplo com:
  - Camada de entrada (2 neurônios)
  - Camada oculta (3 neurônios)
  - Camada de saída (1 neurônio)
- Identifiquem as conexões e indiquem como o feedforward ocorre.

---

## 🧠 Desafio para Casa

**Título:** Criando um Classificador de Portas Lógicas com Feedforward

### Objetivo:
Construir uma rede neural simples (sem backpropagation) que classifique os resultados das portas **AND**, **OR** e **XOR** com base no comportamento observado no feedforward.

### Instruções:
1. Crie um script Python com a classe `NeuralNetwork`.
2. Insira os conjuntos de entrada e saída correspondentes a AND, OR e XOR.
3. Execute o `feedforward()` com diferentes pesos e analise se a rede se comporta corretamente.
4. Explique por que a rede consegue ou não resolver o problema da porta XOR.

### Entrega:
- Código comentado (Python `.ipynb` ou `.py`)
- Tabela de testes
- Pequeno relatório (até 1 página)

---

## ✅ Conclusão

A etapa de **feedforward** é a base do funcionamento das RNAs, sendo essencial para propagar os dados pelas camadas. O **aprendizado** da rede ocorre posteriormente, com o uso do algoritmo de **backpropagation**, que ajusta os pesos com base no erro da saída.

Na próxima aula, veremos como as RNAs **aprendem**, minimizando o erro com **função custo** e **gradiente descendente**.

