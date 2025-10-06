# 🤖 15.2 – Fundamentos da IA Generativa de Perguntas e Respostas

## 🎯 Objetivo

Compreender **como funciona uma IA generativa**, o que é um **modelo de linguagem (LLM)** e como criar um **primeiro sistema de perguntas e respostas (Q&A)** usando modelos pré-treinados.
Ao final, o aluno entenderá **as camadas principais** de um chatbot inteligente e **terá um pipeline básico funcional**.

---

## 🧩 1. O que é uma IA Generativa?

A **IA Generativa** é uma área da Inteligência Artificial capaz de **criar novos conteúdos** — texto, imagens, som, código, entre outros — a partir de dados aprendidos.
Ela **não copia**, mas **gera** algo **novo** baseado em padrões estatísticos do que foi visto durante o treinamento.

### 🧠 Analogia:

Imagine que você estudou milhares de redações e agora sabe **como as pessoas estruturam ideias**.
Quando alguém pede: “Escreva uma redação sobre o meio ambiente”, você **não copia nenhuma**, mas **gera uma nova** baseada em tudo o que aprendeu.
👉 Isso é o que faz um modelo generativo de texto como o **ChatGPT**, **Flan-T5** ou **GPT-Neo**.

---

## 🧬 2. Como funciona um modelo de linguagem (LLM)?

Os **Modelos de Linguagem de Grande Escala (LLMs)** são redes neurais (normalmente baseadas em **Transformers**) que **aprendem a prever a próxima palavra** em uma frase.
Por exemplo:

> Entrada: “A IA generativa é uma área da…”
> Modelo: “inteligência artificial que cria novos conteúdos.”

### 🔍 Etapas principais:

1. **Tokenização:** transforma o texto em números (ex: “IA” → 4023).
2. **Embedding:** converte esses números em **vetores contínuos**, que representam o significado das palavras.
3. **Treinamento:** a rede aprende padrões de contexto (semântica e sintaxe).
4. **Inferência:** o modelo gera novas palavras baseando-se na probabilidade do contexto.

---

## ⚙️ 3. Arquitetura de um sistema de Perguntas e Respostas

Um sistema Q&A (Question Answering) usa um modelo pré-treinado para encontrar a **resposta mais provável** dentro de um **contexto fornecido**.

### 🔄 Pipeline Simplificado:

```
Pergunta → [Processamento] → Modelo Q&A → [Extração de resposta]
```

Exemplo:

> **Contexto:** “A IA generativa cria conteúdo novo a partir de dados existentes.”
> **Pergunta:** “O que a IA generativa faz?”
> **Resposta:** “Cria conteúdo novo a partir de dados existentes.”

### 💡 Analogia:

Pense no modelo como um **aluno que leu um livro inteiro (contexto)**.
Quando você faz uma pergunta, ele **procura a resposta dentro do livro**, usando o que aprendeu sobre a linguagem.

---

## 🧠 4. Tipos de Modelos Q&A

| Tipo           | Exemplo                              | Funcionamento                                      |
| -------------- | ------------------------------------ | -------------------------------------------------- |
| **Extractivo** | BERT, DistilBERT                     | Extrai a resposta literal do contexto              |
| **Generativo** | FLAN-T5, GPT-Neo                     | Gera uma nova resposta com base no contexto        |
| **Híbrido**    | RAG (Retrieval-Augmented Generation) | Busca informações + gera a resposta com base nelas |

### ⚖️ Comparando os dois primeiros:

* **BERT** → funciona como um “caçador de frases” dentro de um texto.
* **FLAN-T5 / GPT** → funciona como um “escritor inteligente” que lê o contexto e **reformula** a resposta.

---

## 🧪 5. Mão na Massa – Criando seu primeiro Q&A

Vamos usar o **Hugging Face Transformers**, que fornece modelos pré-treinados de fácil uso.

### 🔹 Instalação

```bash
pip install transformers
```

### 🔹 Implementação

```python
from transformers import pipeline

# Criar um pipeline de Perguntas e Respostas
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Contexto (texto base)
context = """
A inteligência artificial generativa é uma área da IA que cria novos conteúdos,
como textos, imagens e sons, a partir de padrões aprendidos em grandes conjuntos de dados.
"""

# Pergunta
question = "O que é inteligência artificial generativa?"

# Modelo responde
result = qa_model(question=question, context=context)
print("Resposta:", result["answer"])
print("Pontuação de confiança:", result["score"])
```

### 🔍 Explicando cada parte:

* `pipeline("question-answering")` → cria um processo especializado em Q&A.
* `model="distilbert-base-cased-distilled-squad"` → usa um modelo pré-treinado da base SQuAD (Stanford Q&A Dataset).
* `context` → é o “livro” que o modelo vai ler.
* `question` → é a pergunta feita pelo usuário.
* `result["answer"]` → é a resposta que o modelo extraiu.
* `result["score"]` → é a **confiança** do modelo (0 a 1).

### 🔁 Exemplo de saída:

```
Resposta: cria novos conteúdos
Pontuação de confiança: 0.96
```

---

## 🔍 6. Explorando e modificando o contexto

Testar o modelo com diferentes contextos e observar como a resposta muda é uma parte essencial do aprendizado.

### Experimento:

```python
context = """
O aprendizado profundo é uma área da inteligência artificial que utiliza redes neurais
para aprender representações complexas dos dados.
"""

question = "O que é inteligência artificial generativa?"
```

O modelo agora **não encontrará a resposta**, pois **o contexto não a contém**.
👉 Isso ensina aos alunos que o **modelo de Q&A não “sabe tudo”** — ele **depende do contexto fornecido.**

### 🧩 Conclusão parcial:

Modelos como o **DistilBERT** **não “inventam” respostas** — eles **buscam trechos existentes**.
Para gerar respostas novas (como o ChatGPT faz), precisaremos de um **modelo generativo**, que aprenderemos nas próximas semanas (FLAN-T5, GPT-Neo, etc.).

---

## 📊 7. Experimento Complementar: Q&A em Português

```python
qa_model_pt = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese")

context = """
A IA generativa é uma área da inteligência artificial que permite a criação de textos, imagens e sons.
Ela é usada em educação, arte, engenharia e entretenimento.
"""

question = "Em quais áreas a IA generativa é usada?"
result = qa_model_pt(question=question, context=context)
print("Resposta:", result["answer"])
```

💬 **Resultado esperado:**

```
educação, arte, engenharia e entretenimento
```

Agora o modelo responde **em português** — excelente para aplicações educacionais.

---

## 💡 8. Atividade Prática (em grupo)

### **Título:** Criando seu primeiro Assistente de Perguntas e Respostas

**Objetivo:** Implementar um mini chatbot que responda perguntas sobre um tema de sua escolha.

### **Etapas:**

1. Escolha um **tema** (ex.: energia renovável, Python, história da IA).
2. Crie um **contexto textual** com 4–5 parágrafos.
3. Utilize o modelo `distilbert-base-cased-distilled-squad` (ou em português).
4. Teste **5 perguntas** diferentes.
5. Registre os resultados e a **pontuação de confiança**.
6. Discuta: o modelo entendeu todas as perguntas corretamente?

### **Entrega:**

* Notebook `.ipynb` ou script `.py`
* Documento com:

  * Tema e contexto
  * Perguntas testadas
  * Respostas obtidas
  * Observações do grupo

---

## 📘 9. Referências

* Vaswani et al. (2017). *Attention is All You Need.* NeurIPS.
* Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL.
* Cho et al. (2014). *Learning phrase representations using RNN encoder–decoder for statistical machine translation.* EMNLP.
* Hugging Face Docs: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
