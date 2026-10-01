# Detecção de Fraude em Cartão de Crédito

Este repositório contém um projeto de detecção de transações fraudulentas em cartões de crédito .

---

##  O Problema e o Desbalanceamento
Em cenários de detecção de fraude financeira, a proporção de transações ilegítimas é extremamente baixa (~0,17%). 

* **Por que a acurácia engana?** Um modelo simplista que prevê que *todas* as transações são legítimas alcança **99,83% de acurácia**, mas deixa passar 100% das fraudes, resultando em grande prejuízo financeiro.
* **Métricas Principais:** O foco da avaliação está no **Recall** (capacidade de capturar as fraudes existentes), na **Precisão** (redução de alarmes falsos) e no **F1-Score** (equilíbrio entre ambos) para a classe minoritária.

---

##  Preparação dos Dados
1. **Engenharia de Atributos:** Transformação logarítmica da variável `Amount` para suavizar a assimetria do valor das transações.
2. **Escalonamento:** Aplicação do `StandardScaler` separadamente para os atributos `Time` e `log_amount`.
3. **Divisão Treino/Teste:** Amostragem estratificada (`stratify=y`) na proporção 80/20 para garantir a mesma porcentagem de fraudes nos dois conjuntos.

---

##  Comparação de Modelos

Abaixo estão os resultados obtidos no conjunto de teste com limiar padrão ($0,5$):

| Modelo | Recall (Fraude) | Precisão (Fraude) | F1-Score |
| :--- | :---: | :---: | :---: |
| **Regressão Logística** *(Baseline)* | ~0,898 | ~0,061 | ~0,114 |
| **Random Forest** *(class_weight='balanced')* | ~0,755 | ~0,937 | ~0,836 |
| **Random Forest + SMOTE** | ~0,806 | ~0,878 | ~0,840 |
| **XGBoost** *(scale_pos_weight)* | ~0,816 | ~0,880 | ~0,847 |

---

##  Otimização do Limiar de Decisão (*Threshold Tuning*)
Com o modelo **XGBoost**, ajustamos o limiar de decisão a partir da curva **Precision-Recall**:

* **Limiar Padrão:** $0,5000$
* **Limiar Otimizado:** Otimizado via F1-Score
* **Resultado:** Aumentou a sensibilidade do modelo capturando mais fraudes sem elevar expressivamente a taxa de falsos positivos.

---

##  Explicabilidade com SHAP
Utilizando o **SHAP (SHapley Additive exPlanations)** no XGBoost, identificamos quais variáveis possuem maior peso nas decisões:

1. **Variáveis mais impactantes:** As características `V14`, `V10`, `V12` e `V17` apresentaram as maiores magnitudes de valor SHAP.
2. **Padrão de Fraude:** Valores extremamente baixos nessas variáveis atreladas a padrões na escala temporal elevam drasticamente a probabilidade calculada pelo modelo.

---

##  Como Executar
1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   ```
2. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
   ```
3. Execute o script ou abra o notebook no Jupyter / Google Colab:
   ```bash
   python credit_card_fraud_detection.py
   ```
