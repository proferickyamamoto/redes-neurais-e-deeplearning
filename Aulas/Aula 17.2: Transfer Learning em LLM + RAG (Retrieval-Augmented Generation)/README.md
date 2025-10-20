# Aula 17.2: Transfer Learning em LLM + RAG (Retrieval-Augmented Generation)

## 🎯 Objetivos

* Entender os princípios de **Transfer Learning** em **LLMs**, por que ele reduz custo e dados.
* Implementar **fine-tuning eficiente** (LoRA/PEFT) em um modelo **text-to-text** (FLAN-T5).
* Compreender o **paradigma RAG** e quando utilizá-lo.
* Construir um pipeline **RAG**: **embeddings → índice vetorial (FAISS) → recuperação → geração**.

---

## 📘 Parte 1 — Teoria: Transfer Learning em LLMs

LLMs (Large Language Models) são pré-treinados em corpora massivos para aprender padrões gerais de linguagem. **Transfer Learning** reutiliza esse conhecimento base e o **adapta** a uma tarefa específica (ex.: QA jurídico, sumarização científica) com **poucos dados e menos tempo**. Duas abordagens comuns:

1. **Fine-tuning completo**: atualiza todos os parâmetros do modelo (caro em memória/tempo).
2. **Fine-tuning eficiente (parameter-efficient)**: congela a maior parte do modelo e treina **camadas adicionais leves** (ex.: **LoRA**, **Adapters**, **Prompt Tuning**). Na prática atual, **LoRA/PEFT** é padrão por reduzir ordens de grandeza de custo e viabilizar treino em GPU única.

Intuição: pense no LLM como um **poliglota experiente**. Em vez de reaprender “o idioma do zero”, você dá **aulas particulares** focadas na **gíria** e **jargão** da sua área (domínio), ajustando apenas “poucos neurônios” — isso é o LoRA.

**Leituras úteis:**

* Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019 (Adapters).
* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
* Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, JMLR 2020.

---

## 💻 Parte 2 — Fine-tuning eficiente (LoRA/PEFT) com FLAN-T5

### Visão

Vamos adaptar **FLAN-T5-base** em um subtarefa de **QA estilo instrução** (entrada: *“Contexto … Pergunta … Responda …”*). Usaremos **PEFT** (Hugging Face) para aplicar **LoRA** às camadas de atenção. O dataset de exemplo será simplificado para fins didáticos (você pode trocar por SQuAD, TyDiQA, ou conjunto proprietário).

### Dependências

```bash
pip install transformers datasets peft accelerate sentencepiece
```

### Código — carregamento do modelo e preparação

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# 1) Base: FLAN-T5
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2) Configuração LoRA (PEFT)
peft_cfg = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,              # rank do LoRA (trade-off custo x qualidade)
    lora_alpha=16,    # escala
    lora_dropout=0.05 # regularização
)
model = get_peft_model(base_model, peft_cfg)

# 3) Dataset de exemplo (substitua por SQuAD ou seu corpus)
data = load_dataset("squad", split={"train":"train[:2%]","validation":"validation[:2%]"})
# Função de formatação: contexto + pergunta -> alvo = resposta curta
def format_example(ex):
    prompt = f"Contexto: {ex['context']}\nPergunta: {ex['question']}\nResponda de forma objetiva:"
    # usa a primeira resposta disponível
    answer = ex["answers"]["text"][0] if len(ex["answers"]["text"])>0 else ""
    return {"prompt": prompt, "answer": answer}

proc_train = data["train"].map(format_example)
proc_val   = data["validation"].map(format_example)

# 4) Tokenização
max_src, max_tgt = 512, 64
def tokenize(batch):
    model_inputs = tokenizer(batch["prompt"], max_length=max_src, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["answer"], max_length=max_tgt, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tok_train = proc_train.map(tokenize, batched=True, remove_columns=proc_train.column_names)
tok_val   = proc_val.map(tokenize, batched=True, remove_columns=proc_val.column_names)

collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

**Explicação:** aplicamos **LoRA** nas camadas de atenção do T5; congelamos o restante. O *dataset* foi minimizado para rodar rápido em aula. Em produção, use todo o conjunto e valide bem.

### Treinando apenas os parâmetros LoRA

```python
args = TrainingArguments(
    output_dir="flan_t5_lora_qa",
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    data_collator=collator,
    tokenizer=tokenizer
)

trainer.train()
```

### Inferência (pós-treino)

```python
def generate_answer(context, question, max_new_tokens=64):
    prompt = f"Contexto: {context}\nPergunta: {question}\nResponda de forma objetiva:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Exemplo
ctx = "A retropropagação ajusta os pesos de uma rede neural minimizando o erro."
q   = "O que é retropropagação?"
print(generate_answer(ctx, q))
```

---

## 📘 Parte 3 — Teoria: RAG (Retrieval-Augmented Generation)

**Problema:** LLMs não “sabem tudo” do seu domínio e podem alucinar. **RAG** combina **recuperação de conhecimento** (embeddings + busca semântica) com **geração condicionada**. Fluxo:

1. **Embeddings**: converter documentos em vetores (Sentence-Transformers).
2. **Índice**: FAISS ou similar.
3. **Consulta**: dado o *prompt*, recuperar **top-k** trechos relevantes.
4. **Geração**: concatenar o contexto recuperado ao *prompt* e pedir a resposta ao LLM (o seu **modelo adaptado via LoRA**, por exemplo).

Analogia: o LLM é um **redator** talentoso; o RAG é a **biblioteca** que você consulta antes de escrever. Juntos: respostas **contextualizadas, atualizadas e auditáveis**.

**Leituras (não-Wikipedia):**

* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP*, NeurIPS 2020.
* Karpukhin et al., *Dense Passage Retrieval (DPR)*, EMNLP 2020.

---

## 💻 Parte 4 — Implementação RAG (FAISS + Embeddings + FLAN-T5)

### Dependências

```bash
pip install sentence-transformers faiss-cpu
```

### Passo 1 — Criar base de conhecimento e embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np, faiss

# 1) Documentos (troque por seus PDFs/HTML já chunkados)
docs = [
    "Gradiente descendente é um método de otimização para minimizar funções.",
    "A função ReLU é muito usada em redes profundas por mitigar saturação.",
    "Batch normalization estabiliza a distribuição das ativações e acelera o treino.",
    "A regularização L2 penaliza pesos grandes e ajuda a reduzir overfitting."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

dim = doc_vecs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_vecs)
```

**Explicação:** usamos um **encoder** de sentenças para capturar semântica. **Normalizamos** para que o **produto interno ≈ cosseno**.

### Passo 2 — Recuperar contexto para uma pergunta

```python
def retrieve(question, k=2):
    qv = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv, k)
    ctx = "\n".join([docs[i] for i in I[0]])
    return ctx, (D[0], I[0])

question = "Como reduzir overfitting em redes neurais?"
context, meta = retrieve(question, k=2)
print("Contexto recuperado:\n", context)
```

### Passo 3 — Geração condicionada ao contexto (RAG “simples”)

Use **o seu modelo LoRA** (adaptado no início) para responder **com base no contexto recuperado**.

```python
def rag_answer(question, max_new_tokens=96):
    context, _ = retrieve(question, k=2)
    prompt = (
        "Use apenas o contexto dado para responder com objetividade.\n"
        f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True), context

ans, used = rag_answer("Como reduzir overfitting em redes neurais?")
print("Resposta:", ans)
print("\n[Contexto usado]\n", used)
```

**Explicação:** essa versão “clássica” do RAG concatena top-k contextos ao *prompt*. Em produção, você pode:

* Resumir os contextos (re-rank, compressão).
* Guardar **metadados** (origem/página/URL).
* Trocar FAISS Flat por **IVF/HNSW** para escalar.

---

## 🧪 Atividade em Sala

1. Substitua o mini-dataset por **SQuAD** (ou base própria PT-BR).
2. Treine **LoRA** por 1–2 épocas (subset) e salve os adaptadores.
3. Construa uma **base vetorial** com **10–20 textos** do seu domínio (ou PDF chunkado).
4. Compare: **(a) sem RAG** vs. **(b) com RAG** (mesmo modelo), usando 5 perguntas do tema.
5. Registre acertos, exemplos e **quando o RAG ajudou**.

---

## 🧠 Desafio para Casa

* Implementar **re-rank** (ex.: `cross-encoder` para reordenar top-k).
* Medir **latência**: tempo da recuperação + geração.
* Adicionar **citações**: ao responder, listar as fontes/trechos usados (IDs dos chunks).
* (Opcional) Interface **Gradio/Streamlit** com campo de upload de PDF → *embed* → RAG.

---

## ✅ Conclusões

* **Transfer Learning** com **LoRA/PEFT** torna o ajuste de LLMs **viável** com poucos recursos.
* **RAG** complementa o LLM com **conhecimento atualizado**, reduz alucinações e dá **traçabilidade** (você sabe de onde veio a resposta).
* A combinação **LoRA + RAG** é hoje um padrão de engenharia para chatbots de domínio.
