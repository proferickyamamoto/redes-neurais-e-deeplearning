# 🧠 Aula 06.1 - Redes Neurais com Memória RAM (RNA RAM)

## 🎯 Objetivo
Apresentar o conceito de Redes Neurais RAM, seu funcionamento, aplicações, vantagens e limitações. Além disso, aplicar o conhecimento por meio de uma implementação prática e uma atividade em sala.

---

## 💾 O que são Redes Neurais RAM?

As **Redes Neurais RAM** são um tipo especial de rede neural que utiliza o conceito de **endereçamento direto de memória**, substituindo o uso de pesos sinápticos por um mapeamento binário simples.

### Características:
- Aprendizado rápido (em uma única exposição).
- Entradas são binárias.
- As RAMs armazenam padrões diretamente na memória.
- Cada RAM recebe um subconjunto das entradas.

---

## 🔄 Funcionamento Geral

### 🧠 Etapas:
1. A entrada binária é dividida em grupos de bits.
2. Cada grupo é usado para endereçar uma célula de RAM.
3. A RAM retorna 1 se aquele endereço foi ativado no treinamento.
4. O somatório das RAMs define a classe da entrada.

### 🎯 Exemplo de Funcionamento:
```
Entrada:  [1, 0, 1, 1, 0, 0]
Divisão:  [1,0] [1,1] [0,0]  → (3 RAMs)
RAM1 → endereço '10' ativado?
RAM2 → endereço '11' ativado?
RAM3 → endereço '00' ativado?
```

---

## 🧮 Ilustração Gráfica

```
[ x1 x2 x3 x4 x5 x6 ]   <- Vetor de entrada binário
  |  |  |  |  |  |
  |  |  └────────────┐
  |  └────┐          │
  ↓       ↓          ↓
[RAM1]  [RAM2] ... [RAMn]   <- Cada RAM armazena bits de entrada (endereçamento)
   ↓       ↓         ↓
  [ Discriminador (soma) ]
           ↓
     Classe prevista
```

---

## ✅ Vantagens e Limitações

### ✅ Vantagens:
- Treinamento extremamente rápido.
- Simplicidade de implementação.
- Boa performance em padrões visuais binários.

### ⚠️ Limitações:
- Requer entradas discretas e binárias.
- Baixa capacidade de generalização.

---

## 💻 Implementação em Python

```python
import numpy as np

class SimpleRAMNeuron:
    def __init__(self):
        self.memory = set()

    def train(self, X):
        for entrada in X:
            chave = tuple(entrada)
            self.memory.add(chave)

    def predict(self, x):
        return 1 if tuple(x) in self.memory else 0

# Exemplo:
entradas_doente = [[1, 0, 1], [1, 1, 1]]
entradas_saudavel = [[0, 0, 0], [0, 1, 0]]

ram_doente = SimpleRAMNeuron()
ram_saudavel = SimpleRAMNeuron()

ram_doente.train(entradas_doente)
ram_saudavel.train(entradas_saudavel)

# Testando nova entrada
paciente = [1, 0, 1]
print("Doente?", ram_doente.predict(paciente))
print("Saudável?", ram_saudavel.predict(paciente))
```

---

## 🧪 Atividade em Sala

**Objetivo:** Aplicar os conceitos de RNA RAM para classificar pacientes com base em sintomas binários.

### Tabela de Dados:
| Febre | Tosse | Dor muscular | Diagnóstico |
|-------|-------|---------------|-------------|
| sim   | não   | sim           | doente      |
| não   | sim   | não           | saudável    |
| sim   | sim   | sim           | doente      |
| não   | não   | não           | saudável    |

### Etapas:
1. Converter as entradas para binário (sim = 1, não = 0).
2. Treinar uma RAM para cada classe.
3. Testar com novos pacientes.

---

## 🧠 Desafio para Casa

**Título:** Classificando Imagens 3x3 com RNA RAM

### Objetivo:
Treinar uma rede RAM para reconhecer imagens binárias 3x3 como letras A ou B.

### Instruções:
- Criar dois conjuntos de padrões (A e B).
- Implementar a RAM com base nesses dados.
- Testar o reconhecimento de padrões conhecidos e modificados.

---

## 📌 Conclusão

As Redes Neurais RAM são uma abordagem poderosa e leve para tarefas com padrões binários bem definidos. Apesar da limitação de generalização, são úteis em sistemas embarcados e aplicações rápidas. A seguir, estudaremos como redes neurais clássicas aprendem com erro, por meio do algoritmo **backpropagation**.

