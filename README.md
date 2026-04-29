# 🩺 Análise e Predição de Câncer de Mama com Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Descrição do Projeto

Este projeto implementa um modelo de **Machine Learning** para classificar tumores de mama como **benignos** ou **malignos** utilizando **Support Vector Machine (SVM)**. O objetivo é auxiliar no diagnóstico precoce de câncer de mama através da análise de características celulares.

## 🎯 Objetivo

Desenvolver um modelo preditivo robusto e confiável que possa identificar corretamente casos de tumores malignos com alta precisão, minimizando falsos negativos (casos críticos onde um tumor maligno é classificado incorretamente como benigno).

## 📊 Dataset

### Breast Cancer Wisconsin Dataset

O dataset utilizado é o **Breast Cancer Wisconsin**, disponível nativamente no scikit-learn. Ele contém dados de 569 amostras de tumores, coletados através de **punção aspirativa por agulha fina (FNA)** de massas mamárias.

#### Características do Dataset:

- **Total de Amostras**: 569
- **Total de Features**: 30 características numéricas
- **Classes**:
  - **0 - Maligno**: 212 casos (37.26%)
  - **1 - Benigno**: 357 casos (62.74%)

#### Features Principais:

Para cada célula, foram calculadas 10 características principais e suas estatísticas (média, erro padrão e "pior" valor):

1. **Radius** (raio): distância média do centro aos pontos do perímetro
2. **Texture** (textura): desvio padrão dos valores de escala de cinza
3. **Perimeter** (perímetro): tamanho do núcleo
4. **Area** (área): área do núcleo
5. **Smoothness** (suavidade): variação local nos comprimentos do raio
6. **Compactness** (compacidade): (perímetro² / área) - 1.0
7. **Concavity** (concavidade): severidade das porções côncavas do contorno
8. **Concave points** (pontos côncavos): número de porções côncavas do contorno
9. **Symmetry** (simetria): simetria do núcleo
10. **Fractal dimension** (dimensão fractal): "aproximação da linha costeira" - 1

### Importância Clínica

Este dataset é amplamente utilizado na comunidade científica para:
- Desenvolvimento de sistemas de apoio à decisão médica
- Pesquisa em diagnóstico automatizado de câncer
- Benchmarking de algoritmos de classificação

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Computação numérica
- **Matplotlib**: Visualização de dados
- **Seaborn**: Visualização estatística avançada
- **Scikit-learn**: Algoritmos de Machine Learning

## 📈 Metodologia

### 1. Exploração dos Dados (EDA)
- Análise estatística descritiva
- Visualização de correlações entre features
- Análise de distribuição das classes
- Identificação de padrões e outliers

### 2. Pré-processamento
- Separação de features (X) e target (y)
- Divisão em conjuntos de treino (80%) e teste (20%) com `stratify=y`
- **Normalização Min-Max** ajustada apenas no treino e aplicada no teste
- Uso de **Pipeline** no Grid Search para evitar vazamento de dados (data leakage)

### 3. Modelagem
Três abordagens foram testadas:

#### a) SVM Inicial (sem normalização)
- Modelo baseline para comparação
- Parâmetros padrão do scikit-learn

#### b) SVM com Normalização
- Aplicação de Min-Max Scaling
- Melhora significativa na performance

#### c) SVM Otimizado (Grid Search)
- Busca exaustiva dos melhores hiperparâmetros
- Parâmetros testados:
  - **C**: [0.1, 1, 10, 100] - Parâmetro de regularização
  - **gamma**: [1, 0.1, 0.01, 0.001] - Coeficiente do kernel RBF
  - **kernel**: ['rbf'] - Radial Basis Function
- Pipeline: `MinMaxScaler` + `SVC`

### 4. Avaliação
Métricas utilizadas:
- **Acurácia**: Porcentagem de predições corretas
- **Matriz de Confusão**: Análise detalhada de acertos e erros
- **Precision, Recall, F1-Score**: Métricas por classe
- **Validação Cruzada**: 5-fold cross-validation no Grid Search
- **Overfitting**: comparação treino vs teste
- **Falsos Negativos (FN)**: foco na classe maligna
- **Validação Cruzada Estratificada**: 5-fold com `StratifiedKFold`

## 📊 Resultados

### Comparação de Performance

| Modelo | Acurácia | Observação |
|--------|----------|------------|
| SVM Inicial | ~62.8% | Sem normalização |
| SVM Normalizado | ~96.5% | Com Min-Max Scaling |
| SVM Otimizado | ~97.4% | Com Grid Search |

### Confiabilidade (checagens finais)

Resultados da última execução (podem variar levemente):

- **Acurácia teste (modelo otimizado)**: 0.9737
- **Acurácia treino vs teste**: 0.9890 vs 0.9737 (gap pequeno)
- **Falsos Negativos (malignos)**: 2
- **Recall da classe maligna**: 0.9762
- **Balanceamento**: 62.74% benignos / 37.26% malignos

### Validação Cruzada Estratificada (5-fold)

- **Accuracy**: 0.9772 ± 0.0163
- **Precision**: 0.9781 ± 0.0183
- **Recall**: 0.9861 ± 0.0124
- **F1**: 0.9820 ± 0.0126

### Métricas por Classe (teste, modelo otimizado)

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Maligno | 0.9535 | 0.9762 | 0.9647 | 42 |
| Benigno | 0.9859 | 0.9722 | 0.9790 | 72 |

### Matriz de Confusão

A **Matriz de Confusão** é crucial neste contexto médico:

```
                Predito
              Maligno  Benigno
Real Maligno    [TP]    [FN] ⚠️ CRÍTICO
     Benigno    [FP]    [TN]
```

- **Verdadeiros Positivos (TP)**: Tumores benignos corretamente identificados
- **Verdadeiros Negativos (TN)**: Tumores malignos corretamente identificados
- **Falsos Positivos (FP)**: Falso alarme (benigno classificado como maligno)
- **Falsos Negativos (FN)**: **CRÍTICO** - Tumor maligno não detectado

### Por que a Matriz de Confusão é Importante?

1. **Contexto Médico**: Um falso negativo pode ser fatal - não detectar um câncer maligno
2. **Trade-off**: Em aplicações médicas, é preferível ter mais falsos positivos (investigações adicionais) do que falsos negativos
3. **Métricas Derivadas**: Permite calcular Sensibilidade (Recall) e Especificidade
4. **Decisão Clínica**: Ajuda a determinar se o modelo é seguro para uso em triagem

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Execução

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/breast-cancer-prediction.git
cd breast-cancer-prediction
```

2. Abra o Jupyter Notebook:
```bash
jupyter notebook Breast_data.ipynb
```

3. Execute todas as células sequencialmente

## 📁 Estrutura do Projeto

```
Breast Cancer/
│
├── Breast_data.ipynb    # Notebook principal com toda a análise
├── README.md            # Documentação do projeto
└── requirements.txt     # Dependências (opcional)
```

## 🔍 Insights e Conclusões

1. **Normalização é Essencial**: A acurácia aumentou de ~63% para ~97% apenas com normalização
2. **Grid Search otimiza ainda mais**: Ganho adicional de ~1% com hiperparâmetros otimizados
3. **Dataset Balanceado**: Com ~37% malignos e ~63% benignos, não há desbalanceamento severo
4. **Features Correlacionadas**: Muitas features possuem alta correlação (radius, perimeter, area)
5. **SVM é Efetivo**: Para datasets de tamanho médio e features numéricas, SVM é uma excelente escolha

## 🎓 Aprendizados

- **Importância do pré-processamento**: Normalização é crítica para algoritmos baseados em distância
- **Validação rigorosa**: Em aplicações médicas, métricas como Recall são mais importantes que Acurácia
- **Otimização de hiperparâmetros**: Grid Search pode melhorar significativamente o modelo
- **Interpretabilidade**: Matrizes de confusão fornecem insights valiosos sobre o comportamento do modelo

## 🔮 Melhorias Futuras

- [ ] Implementar outros algoritmos (Random Forest, XGBoost, Neural Networks)
- [ ] Análise de importância das features
- [ ] Ajustar limiar de decisão com foco em reduzir falsos negativos
- [ ] Implementar técnicas de explicabilidade (SHAP, LIME)
- [ ] Deploy do modelo em API Flask/FastAPI
- [ ] Interface web para predições em tempo real

## 📚 Referências

- [UCI Machine Learning Repository - Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- Wolberg, W.H., Street, W.N., and Mangasarian, O.L. (1995). "Breast Cancer Wisconsin (Diagnostic) Data Set"

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.

## 👥 Autor

Desenvolvido com 💙 para aprendizado e pesquisa em Machine Learning aplicado à saúde.

---

⭐ **Se este projeto foi útil, considere dar uma estrela!** ⭐

## 📞 Contato

Para dúvidas, sugestões ou colaborações, abra uma issue no repositório.
