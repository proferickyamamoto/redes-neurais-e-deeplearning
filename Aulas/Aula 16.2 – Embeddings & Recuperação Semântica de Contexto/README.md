# Aula 16.2 – Embeddings & Recuperação Semântica de Contexto

## 🎯 Objetivo

Nesta semana os alunos vão entender:

* O que são embeddings de frase e porque são úteis para recuperar contexto relevante.
* Como construir uma base vetorial de documentos (conhecimento) e consultá-la eficientemente via busca semântica (ex: com FAISS).
* Como usar essa base vetorial dentro do pipeline de Q&A para melhorar as respostas do modelo generativo.

---

## 📘 Parte 1 – Teoria: embeddings de frase e similaridade semântica

### 🔹 O que são embeddings de frase?

Embeddings de frase (sentence embeddings) transformam uma frase inteira em um vetor numérico fixo que captura seu significado semântico (contexto, sinônimos, intenções). Diferentemente de embeddings de palavras (Word2Vec, GloVe), eles permitem comparar sentenças completas. ([Wikipedia][1])

Por exemplo, as frases “O gato está dormindo” e “O felino repousa” devem gerar embeddings próximos, porque expressam ideias semelhantes.

Modelos como **Sentence-BERT (SBERT)** otimizam embeddings de frase usando redes siamesas (Siamese Networks) para treinar de modo que frases semanticamente próximas fiquem próximas no espaço vetorial. ([sbert.net][2])

### 🔹 Métricas de similaridade

Para comparar embeddings, usamos métricas como:

* **Distância Euclidiana (L2)**: mede a “distância padrão” entre vetores.
* **Produto interno (dot product) / Similaridade cosseno**: considera direção dos vetores; útil para embeddings normalizados.

A similaridade ajuda a encontrar quais documentos são semanticamente mais próximos à pergunta.

### 🔹 Biblioteca FAISS para busca eficiente

Quando temos muitos embeddings (milhares ou milhões), buscar exaustivamente é caro. A FAISS (Facebook AI Similarity Search) é uma biblioteca otimizada em C++ com bindings Python para indexar vetores densos e realizar buscas de vizinhos mais próximos de forma eficiente. ([Engineering at Meta][3])

Por exemplo, você pode adicionar embeddings ao índice FAISS e, dado um vetor de consulta, recuperar os k documentos mais similares em tempo rápido.

FAISS suporta diferentes tipos de índices (flat, IVFFlat, HNSW, etc.) para escalabilidade e precisão. ([Medium][4])

---

## 💻 Parte 2 – Implementação prática: pipeline de vetorização + indexação + busca

Aqui vamos construir:

1. Uma base de documentos (contextos) do conhecimento.
2. Gerar embeddings de frase para cada documento.
3. Construir índice FAISS para busca.
4. Dada uma pergunta, embedar a pergunta e recuperar os documentos mais relevantes.

### 🔹 Instalação de dependências

```bash
pip install sentence-transformers faiss-cpu
```

(O `faiss-cpu` é a versão sem GPU. Se você tiver GPU, use `faiss-gpu`.)

### 🔹 Código: vetorização e indexação

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Definir documentos do conhecimento
docs = [
    "A IA generativa cria novos conteúdos a partir de dados existentes.",
    "Redes neurais aprendem padrões complexos dos dados.",
    "Embeddings transformam texto em vetores numéricos.",
    "Transformers são a arquitetura central em modelos de linguagem.",
    "Busca semântica melhora a recuperação de contexto relevante."
]

# 2. Gerar embeddings com modelo pré-treinado
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model_embed.encode(docs, convert_to_numpy=True)
# embeddings.shape = (n_docs, dim_embedding)

# 3. Construir índice FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # índice para similaridade por produto interno
faiss.normalize_L2(embeddings)  # normalizar vetores (para usar produto interno como cosseno)
index.add(embeddings)
```

### 🔹 Código: buscar contexto relevante para uma pergunta

```python
def retrieve_top_k(question, k=2):
    q_emb = model_embed.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)  # D: distâncias (scores), I: índices dos documentos
    return I[0], D[0]

# Exemplo:
q = "O que é IA generativa?"
idxs, scores = retrieve_top_k(q, k=2)
print("Documentos mais relevantes:")
for i, score in zip(idxs, scores):
    print(f"- {docs[i]} (score {score:.4f})")
```

Explicações:

* `IndexFlatIP(dim)` cria um índice que compara vetores via produto interno (útil quando embeddings já estão normalizados).
* `normalize_L2(...)` faz a normalização para que dot product equivalha à similaridade cosseno.
* `index.search(...)` retorna os k documentos mais similares ao vetor da pergunta.

### 🔹 Integração com pipeline de Q&A

Depois de recuperar os documentos mais relevantes (por exemplo, os top-2), você pode concatená-los como `context` para o modelo de perguntas-respostas ou modelo generativo:

```python
context = " ".join(docs[i] for i in idxs)  # juntar os documentos recuperados
# Então alimentamos o pipeline de Q&A ou modelo generativo com esse contexto
```

Isso permite que o modelo tenha acesso ao trecho mais relevante do “conhecimento” para responder.

---

## 🧠 Parte 3 – Exemplos e analogias para entendimento

### 🧩 Analogia de busca semântica

Imagine que cada documento é uma **estrela** em um mapa galáctico (o espaço vetorial).
Quando você faz uma pergunta, você gera uma **estrela consulta** nesse mapa.
A FAISS ajuda você a encontrar qual estrela/documento está mais próxima da sua estrela de pergunta — essas estrelas próximas representam documentos semanticamente mais parecidos.

### ⚖️ Por que FAISS é mais eficiente?

Se você tivesse que comparar manualmente cada documento da base com a pergunta (busca exaustiva), isso custaria muito tempo.
FAISS usa estruturas de índice (listas invertidas, quantização, grafos de vizinhança) para **reduzir drasticamente o número de comparações**, suportando buscas rápidas. ([Medium][4])

---

## 🧪 Atividade em Sala – Recuperação Contextual

**Título:** Construindo e utilizando uma base vetorial de contexto

### Instruções:

1. Use a base de documentos `docs` ou crie sua própria base (5 a 10 parágrafos).
2. Gere embeddings de frase usando **SentenceTransformer** (ex: `all-MiniLM-L6-v2`).
3. Indexe com FAISS e normalize os embeddings.
4. Para 3 perguntas distintas, recupere os top-k documentos relevantes.
5. Exiba os documentos, scores e discuta se as escolhas fazem sentido.

### Resultados esperados:

* Pergunta “O que é IA generativa?” → deve recuperar o documento “A IA generativa cria novos conteúdos…” com score alto.
* Pergunta “Como funcionam embeddings?” → deve recuperar “Embeddings transformam texto em vetores…” e possivelmente “Redes neurais aprendem …”
* Pergunta “O que são transformers?” → deve recuperar “Transformers são a arquitetura central…”

Explique por que a busca deu esses resultados.

---

## 📄 Desafio para Casa – Vetorizar base maior + testar pipeline completo

**Título:** Q&A com base de contexto dinâmica

### Instruções:

1. Reúna 20–30 textos sobre um tema escolhido (ex: “Redes Neurais e Deep Learning”).
2. Gere embeddings e construa índice FAISS (ou use variante aproximada).
3. Para cada pergunta da turma (ex: 5 perguntas), recupere top-k documentos.
4. Use o modelo de perguntas-respostas (da Semana 1 ou modelo generativo) com esse contexto recuperado.
5. Compare: respostas com contexto recuperado vs. respostas com contexto fixo (sem indexação).
6. Entregue notebook + relatório técnico (incluindo nomes/RMs).

---

## 📚 Referências Relevantes

* Sentence Transformers: biblioteca prática para embeddings de frase. ([sbert.net][2])
* FAISS: biblioteca eficiente para busca de similaridade vetorial. ([Engineering at Meta][3])
* “Top 4 Sentence Embedding Techniques” — introdução a embeddings de frase. ([Analytics Vidhya][5])


[1]: https://en.wikipedia.org/wiki/Sentence_embedding?utm_source=chatgpt.com "Sentence embedding"
[2]: https://sbert.net/?utm_source=chatgpt.com "SentenceTransformers Documentation — Sentence Transformers ..."
[3]: https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/?utm_source=chatgpt.com "Faiss: A library for efficient similarity search - Engineering at Meta"
[4]: https://medium.com/%40devbytes/similarity-search-with-faiss-a-practical-guide-to-efficient-indexing-and-retrieval-e99dd0e55e8c?utm_source=chatgpt.com "Similarity Search with FAISS: A Practical Guide to Efficient Indexing ..."
[5]: https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/?utm_source=chatgpt.com "Top 4 Sentence Embedding Techniques using Python"
