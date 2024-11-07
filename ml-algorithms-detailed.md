# Algoritmos de Machine Learning: Uma Explicação Detalhada

## 1. Regressão Linear

### O Que É?
A regressão linear é um dos algoritmos mais fundamentais em machine learning. Ela busca encontrar uma relação linear entre variáveis, como por exemplo, a relação entre o tamanho de uma casa e seu preço.

### Matemática Por Trás
Para uma regressão linear simples:
$$y = β_0 + β_1x + ε$$

- **β₀**: O intercepto (onde a linha cruza o eixo y)
- **β₁**: A inclinação (quanto y muda quando x aumenta em 1 unidade)
- **ε**: Termo de erro (representa o que nosso modelo não consegue explicar)

### Como Encontrar os Melhores Parâmetros?
Usamos o Método dos Mínimos Quadrados:

1. Primeiro, definimos o erro quadrático:
   $$E = \sum_{i=1}^n (y_i - (β_0 + β_1x_i))^2$$

2. Para minimizar, derivamos em relação a β₀ e β₁ e igualamos a zero:
   $$β_1 = \frac{n\sum x_iy_i - \sum x_i\sum y_i}{n\sum x_i^2 - (\sum x_i)^2}$$
   $$β_0 = \bar{y} - β_1\bar{x}$$

### Exemplo Prático
Para prever preços de casas:
- x: tamanho em m²
- y: preço em reais
- β₀: preço base
- β₁: aumento de preço por m²

## 2. Regressão Logística

### O Que É?
A regressão logística é usada para classificação binária. Por exemplo, prever se um email é spam (1) ou não (0).

### Como Funciona?
1. Primeiro, calculamos uma combinação linear como na regressão linear:
   $$z = β_0 + β_1x_1 + β_2x_2 + ... + β_nx_n$$

2. Depois, aplicamos a função sigmóide para obter uma probabilidade:
   $$P(y=1|x) = \frac{1}{1 + e^{-z}}$$

### A Função Sigmóide
- Transforma qualquer número real em um valor entre 0 e 1
- É suave e diferenciável
- Tem formato em "S"

### Treinamento
Usamos a função de custo de entropia cruzada:
$$J(θ) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)}\log(h_θ(x^{(i)})) + (1-y^{(i)})\log(1-h_θ(x^{(i)}))]$$

## 3. K-Nearest Neighbors (KNN)

### O Que É?
KNN é um algoritmo de classificação/regressão que se baseia na proximidade dos dados.

### Como Funciona?
1. Para um novo ponto x:
   - Calcula a distância para todos os pontos do conjunto de treino
   - Encontra os k pontos mais próximos
   - Toma a decisão baseada nesses k vizinhos

### Cálculo de Distância
**Distância Euclidiana:**
$$d(p,q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$

**Distância de Manhattan:**
$$d(p,q) = \sum_{i=1}^n |p_i - q_i|$$

### Escolha do K
- K pequeno: modelo mais complexo, pode ter overfitting
- K grande: modelo mais simples, pode ter underfitting
- Geralmente usa-se validação cruzada para escolher K

## 4. Support Vector Machines (SVM)

### O Que É?
SVM busca encontrar o hiperplano que melhor separa as classes com a maior margem possível.

### Matemática Fundamental
**Hiperplano:**
$$w^Tx + b = 0$$

**Função Objetivo:**
$$\min_{w,b} \frac{1}{2}||w||^2$$
Sujeito a: $$y_i(w^Tx_i + b) ≥ 1$$

### Kernel Trick
Permite trabalhar em dimensões maiores sem calcular explicitamente a transformação:
$$K(x,y) = ⟨φ(x),φ(y)⟩$$

Kernels comuns:
- Linear: $$K(x,y) = x^Ty$$
- RBF: $$K(x,y) = exp(-\frac{||x-y||^2}{2σ^2})$$

## 5. Árvores de Decisão

### O Que É?
Uma árvore de decisão divide os dados em subgrupos baseado em regras de decisão.

### Como Construir?
1. **Escolha da Melhor Divisão:**
   - Usando Entropia: $$H(S) = -\sum_{i=1}^c p_i \log_2(p_i)$$
   - Ou Índice Gini: $$G(S) = 1 - \sum_{i=1}^c p_i^2$$

2. **Ganho de Informação:**
   $$IG(S,A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|}H(S_v)$$

### Poda da Árvore
Para evitar overfitting:
- Pré-poda: Limitar profundidade, número mínimo de amostras
- Pós-poda: Remover nós que não melhoram significativamente o desempenho

## 6. Random Forest

### O Que É?
Conjunto de árvores de decisão que "votam" para fazer previsões.

### Como Funciona?
1. Cria múltiplas árvores usando:
   - Bagging (Bootstrap Aggregating)
   - Seleção aleatória de features

2. Para classificação:
   - Voto majoritário
   $$classe = \text{modo}(predicoes_{arvore_1}, ..., predicoes_{arvore_n})$$

3. Para regressão:
   - Média das previsões
   $$previsao = \frac{1}{n}\sum_{i=1}^n predicoes_{arvore_i}$$

## 7. Gradient Boosting

### O Que É?
Conjunto de modelos fracos (geralmente árvores) que são treinados sequencialmente para corrigir os erros dos anteriores.

### Processo de Treinamento
1. Treina primeiro modelo nos dados originais
2. Para cada novo modelo:
   - Calcula os resíduos (erros) do modelo atual
   - Treina novo modelo para prever esses resíduos
   - Atualiza previsões: $$F_m(x) = F_{m-1}(x) + γ_mh_m(x)$$

### Variantes Populares
- XGBoost
- LightGBM
- CatBoost

## 8. Redes Neurais

### Estrutura Básica
- **Camada de Entrada:** Recebe os dados
- **Camadas Ocultas:** Processam informações
- **Camada de Saída:** Produz resultado

### Matemática Por Trás
1. **Forward Propagation:**
   $$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
   $$a^{[l]} = g^{[l]}(z^{[l]})$$

2. **Funções de Ativação:**
   - ReLU: $$f(x) = max(0,x)$$
   - Sigmoid: $$f(x) = \frac{1}{1+e^{-x}}$$
   - Tanh: $$f(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$

3. **Backpropagation:**
   $$\frac{∂L}{∂w^{[l]}} = \frac{∂L}{∂a^{[l]}} \frac{∂a^{[l]}}{∂z^{[l]}} \frac{∂z^{[l]}}{∂w^{[l]}}$$

## Dicas Práticas de Implementação

### 1. Pré-processamento
- Normalização dos dados
- Tratamento de valores faltantes
- Codificação de variáveis categóricas

### 2. Validação
- Validação cruzada
- Hold-out set
- Validação temporal para séries temporais

### 3. Métricas de Avaliação
- **Classificação:**
  - Acurácia
  - Precisão
  - Recall
  - F1-Score
  
- **Regressão:**
  - MSE
  - MAE
  - R²

### 4. Hiperparâmetros
- Grid Search
- Random Search
- Validação cruzada para otimização

## Quando Usar Cada Algoritmo?

### Regressão Linear
- Relações lineares simples
- Dataset pequeno
- Necessidade de interpretabilidade

### Regressão Logística
- Classificação binária
- Probabilidades são importantes
- Dataset bem comportado

### KNN
- Dataset pequeno/médio
- Fronteiras de decisão não-lineares
- Fase de treino rápida necessária

### SVM
- Dataset médio
- Dados de alta dimensionalidade
- Fronteiras de decisão complexas

### Árvores de Decisão
- Necessidade de interpretabilidade
- Dados mistos (numéricos/categóricos)
- Regras de decisão claras

### Random Forest
- Dataset grande
- Tolerância a outliers
- Paralelização possível

### Gradient Boosting
- Melhor performance possível
- Tempo de treino não é crítico
- Dataset estruturado

### Redes Neurais
- Dataset muito grande
- Problemas complexos
- Dados não estruturados (imagens, texto)