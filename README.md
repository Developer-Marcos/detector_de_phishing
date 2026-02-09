### Classificador Ensemble com Voting Classifier e GridSearch

Este projeto implementa um sistema de classificação de sites de phishing que combina múltiplos modelos de machine learning usando Voting Classifier com otimização de hiperparâmetros via GridSearchCV. Além disso, o desempenho é avaliado usando validação cruzada e comparado graficamente entre os modelos.

---

### Funcionalidades principais

- Treinamento de 3 modelos base:
  - Decision Tree Classifier (Árvore de Decisão)
  - Random Forest Classifier (Floresta Aleatória)
  - Support Vector Machine (SVM)

- Otimização de hiperparâmetros com GridSearchCV para cada modelo, garantindo melhor performance.

- Combinação dos modelos base usando Voting Classifier (votação soft), com pesos ajustáveis para cada modelo.

- Avaliação do modelo final com:
  - Acurácia no conjunto de teste
  - Relatório detalhado de métricas (precision, recall, f1-score)
  - Validação cruzada (Cross-Validation) para medir robustez e evitar overfitting

- Visualização comparativa das acurácias dos modelos base e do ensemble por meio de gráfico de barras estilizado.

---

### Tecnologias utilizadas

- Python 3.x
- scikit-learn (sklearn)
- NumPy
- Matplotlib
- seaborn

---

### Rodando com Docker (Recomendado)

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

Para facilitar o teste da aplicação sem a necessidade de configurar um ambiente Python local, você pode utilizar o Docker.

###### Assim o projeto inicializa so de rodar a imagem 
Construa a imagem:

```
docker build -t detector-phishing .  
```

Execute o container:

```
docker run -it --rm -e PYTHONUNBUFFERED=1 detector-phishing
```

Assim o projeto roda de forma direta, sem precisar de mais nenhuma config.

---
### Como usar

1. Clone o repositório:

```bash
git clone https://github.com/Developer-Marcos/detector_de_phishing.git
```
2. Entre na pasta do projeto:
 ```bash
cd detector_de_phishing
```  
4. Instale as dependências via ```pip```:
 ```bash
pip install -r requirements.txt
```  
5. Execute o script principal com Python:
 ```bash
python main.py
```  
