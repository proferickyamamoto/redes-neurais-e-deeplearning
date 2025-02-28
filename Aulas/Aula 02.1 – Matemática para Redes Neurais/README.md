# 📌 Fundamentos Matemáticos para Redes Neurais

## 🎯 Objetivos da Aula
Nesta aula, vamos abordar os seguintes tópicos:
- Conceitos fundamentais de **Álgebra Linear** para Redes Neurais.
- Revisão de **Cálculo Diferencial** e sua aplicação no aprendizado de máquinas.
- Introdução ao **Gradiente Descendente**, utilizado para otimização de redes neurais.

---

## 🔢 1. Álgebra Linear para Redes Neurais
A **Álgebra Linear** é essencial para o funcionamento das redes neurais, pois permite manipular grandes conjuntos de dados de maneira eficiente.

### 🟢 **Vetores**
Os vetores representam magnitudes e direções no espaço. Um vetor **v** pode ser escrito como:

$$\ v = \begin{bmatrix} v_1 \ v_2 \dots \ v_n \end{bmatrix} \$$

### 🔵 **Matrizes**
As matrizes armazenam dados organizados em **linhas e colunas**. Exemplo:

$$\ A = \begin{bmatrix} a_{11} & a_{12} \\ 
a_{21} & a_{22} \end{bmatrix} \$$

### ✏️ **Operações Matriciais**
1️⃣ **Soma de Matrizes**  
$$
C = A + B
$$

2️⃣ **Multiplicação Escalar**  
\[
\lambda A = \begin{bmatrix} \lambda a_{11} & \lambda a_{12} \ \lambda a_{21} & \lambda a_{22} \end{bmatrix}
\]

3️⃣ **Produto Matricial**  
Se \( A \) é uma matriz \( m 	imes n \) e \( B \) é uma matriz \( n 	imes p \), então o produto \( C = A \cdot B \) resulta em uma matriz \( m 	imes p \).

4️⃣ **Transposição de Matriz**  
A transposição troca as linhas e colunas de uma matriz:  
\[
A^T = \begin{bmatrix} a_{11} & a_{21} \ a_{12} & a_{22} \end{bmatrix}
\]

---

## 📈 2. Cálculo Diferencial para Redes Neurais
O **Cálculo Diferencial** permite entender como pequenas mudanças nas entradas afetam a saída da rede neural.

### 🟢 **Derivadas**
A derivada mede a **taxa de variação** de uma função. Exemplo:

Seja a função:
\[
f(x) = x^2 + 3x + 2
\]
Sua derivada é:
\[
f'(x) = 2x + 3
\]

### 🔵 **Derivadas Parciais**
Usadas para funções com múltiplas variáveis. Exemplo:

Seja a função:
\[
f(x, y) = x^2 + 3xy + y^2
\]
A derivada parcial em relação a \( x \) é:
\[
rac{\partial f}{\partial x} = 2x + 3y
\]

---

## 🔥 3. Gradiente Descendente e Função de Custo
O **Gradiente Descendente** é um dos algoritmos mais importantes para otimizar redes neurais. Ele ajusta os pesos do modelo de maneira iterativa para minimizar o erro.

### 📌 **Funções de Custo**
A **função de custo** mede a diferença entre a saída prevista e o valor real. Exemplos:

1️⃣ **Erro Quadrático Médio (MSE)**  
\[
MSE = rac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

2️⃣ **Erro Absoluto Médio (MAE)**  
\[
MAE = rac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

3️⃣ **Entropia Cruzada (Cross-Entropy Loss)**  
\[
L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
\]

### ⚡ **Fórmula do Gradiente Descendente**
\[
w = w -  lpha 
abla L(w)
\]
Onde:
- \(  lpha \) é a **taxa de aprendizado**.
- \( 
abla L(w) \) é o **gradiente da função de custo**.

---

## 🛠 4. Implementação Prática
📌 No arquivo **Jupyter Notebook** anexado, temos exemplos práticos de:
- Soma e multiplicação de matrizes em Python.
- Cálculo de derivadas e derivadas parciais.
- Implementação do Gradiente Descendente para otimização de um modelo.

🔗 **Notebook:** [notebooks/matematica_redes_neurais.ipynb](notebooks/matematica_redes_neurais.ipynb)

---

## 📚 5. Recursos Recomendados
📖 **Livros:**
- “Mathematics for Machine Learning” – Marc Peter Deisenroth.
- “Deep Learning” – Ian Goodfellow.

🎥 **Cursos Online:**
- [Coursera - Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning)

---

## 🚀 6. Próximos Passos
Na próxima aula, exploraremos **Análise Exploratória de Dados (EDA) e Pré-processamento** para preparar os dados antes de treinar modelos de redes neurais!

