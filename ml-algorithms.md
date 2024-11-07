# Algoritmos de Machine Learning: Uma Abordagem Matemática

## Índice
1. Regressão Linear
2. Regressão Logística
3. K-Nearest Neighbors (KNN)
4. Support Vector Machines (SVM)
5. Árvores de Decisão
6. Random Forest
7. Gradient Boosting
8. Redes Neurais
9. K-Means Clustering

## 1. Regressão Linear

### Definição Matemática
A regressão linear modela a relação entre uma variável dependente y e uma ou mais variáveis independentes X.

#### Forma Simples
$$y = β_0 + β_1x + ε$$

Onde:
- y é a variável dependente
- x é a variável independente
- β₀ é o intercepto
- β₁ é o coeficiente de inclinação
- ε é o termo de erro

#### Forma Matricial (Múltiplas Variáveis)
$$Y = Xβ + ε$$

### Minimização do Erro (Método dos Mínimos Quadrados)
$$β = (X^TX)^{-1}X^TY$$

## 2. Regressão Logística

### Função Sigmóide
$$σ(z) = \frac{1}{1 + e^{-z}}$$

### Função de Custo
$$J(θ) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)}\log(h_θ(x^{(i)})) + (1-y^{(i)})\log(1-h_θ(x^{(i)}))]$$

## 3. K-Nearest Neighbors (KNN)

### Distância Euclidiana
$$d(p,q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$

### Distância de Manhattan
$$d(p,q) = \sum_{i=1}^n |p_i - q_i|$$

## 4. Support Vector Machines (SVM)

### Hiperplano de Separação
$$w^Tx + b = 0$$

### Função Objetivo
$$\min_{w,b} \frac{1}{2}||w||^2$$

Sujeito a:
$$y_i(w^Tx_i + b) ≥ 1, i = 1,...,n$$

## 5. Árvores de Decisão

### Entropia
$$H(S) = -\sum_{i=1}^c p_i \log_2(p_i)$$

### Ganho de Informação
$$IG(S,A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|}H(S_v)$$

## 6. Random Forest

### Probabilidade de Voto Majoritário
$$P(h) = \sum_{i=\lceil n/2 \rceil}^n \binom{n}{i} p^i(1-p)^{n-i}$$

Onde:
- n é o número de árvores
- p é a probabilidade de cada árvore fazer a previsão correta

## 7. Gradient Boosting

### Função de Perda
$$L(y, F(x)) = \sum_{i=1}^n l(y_i, F(x_i))$$

### Atualização do Modelo
$$F_m(x) = F_{m-1}(x) + γ_mh_m(x)$$

## 8. Redes Neurais

### Função de Ativação ReLU
$$f(x) = \max(0, x)$$

### Backpropagation
$$\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial y_k} \frac{\partial y_k}{\partial net_k} \frac{\partial net_k}{\partial w_{jk}}$$

## 9. K-Means Clustering

### Função Objetivo
$$J = \sum_{j=1}^k \sum_{i=1}^n ||x_i^{(j)} - c_j||^2$$

### Atualização dos Centróides
$$c_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$$

## Exercícios Práticos

1. **Regressão Linear**: Dado um conjunto de dados, calcule manualmente os coeficientes β₀ e β₁.

2. **Regressão Logística**: Implemente a função sigmóide e calcule a probabilidade para z = 2.

3. **KNN**: Calcule a distância euclidiana entre os pontos (1,2,3) e (4,5,6).

4. **SVM**: Para um conjunto de dados 2D, escreva a equação do hiperplano separador.

## Dicas de Implementação

1. Sempre normalize seus dados antes de aplicar os algoritmos
2. Use validação cruzada para avaliar o desempenho
3. Cuide do overfitting usando regularização quando apropriado
4. Teste diferentes hiperparâmetros

## Referências Sugeridas

1. Pattern Recognition and Machine Learning - Christopher Bishop
2. The Elements of Statistical Learning - Hastie, Tibshirani, Friedman
3. Deep Learning - Goodfellow, Bengio, Courville

---

Lembre-se: A escolha do algoritmo depende muito do seu problema específico, dos dados disponíveis e dos requisitos computacionais. Não existe uma solução única para todos os problemas.