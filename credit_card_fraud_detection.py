# ==============================================================================
# Detecção de Fraude em Cartão de Crédito
# Projeto de Aprendizado de Máquina para Identificação de Transações Fraudulentas
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    auc, 
    f1_score, 
    recall_score, 
    precision_score
)
from imblearn.over_sampling import SMOTE
import shap

# Configurações de estilo para os gráficos
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ------------------------------------------------------------------------------
# 1. Carregamento dos Dados
# ------------------------------------------------------------------------------
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
print(f"Baixando e carregando dados de: {DATA_URL}")
df = pd.read_csv(DATA_URL)

print(f"Formato do conjunto de dados: {df.shape}")
print(df.head())

# ------------------------------------------------------------------------------
# 2. Análise Exploratória de Dados (EDA)
# ------------------------------------------------------------------------------
print("\n=== Informações do Dataset ===")
df.info()

class_counts = df['Class'].value_counts()
class_props = df['Class'].value_counts(normalize=True) * 100

print("\n=== Distribuição das Classes ===")
print(f"Transações Normais (0): {class_counts[0]} ({class_props[0]:.3f}%)")
print(f"Transações Fraudulentas (1): {class_counts[1]} ({class_props[1]:.3f}%)")

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette=['#1f77b4', '#d62728'])
plt.title('Distribuição das Classes (0: Normal, 1: Fraude)', fontsize=14)
plt.xlabel('Classe')
plt.ylabel('Quantidade de Transações')
plt.yscale('log')
plt.show()

# ------------------------------------------------------------------------------
# 3. Pré-processamento e Engenharia de Recursos
# ------------------------------------------------------------------------------
df_prep = df.copy()

# Transformação logarítmica e escalonamento
df_prep['log_amount'] = np.log(df_prep['Amount'] + 0.01)

scaler_time = StandardScaler()
scaler_amount = StandardScaler()

df_prep['scaled_time'] = scaler_time.fit_transform(df_prep[['Time']])
df_prep['scaled_log_amount'] = scaler_amount.fit_transform(df_prep[['log_amount']])

# Remoção das colunas originais
df_prep.drop(columns=['Time', 'Amount', 'log_amount'], inplace=True)

X = df_prep.drop(columns=['Class'])
y = df_prep['Class']

# Divisão Treino/Teste estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTamanho do treino: {X_train.shape[0]} | Fraudes no treino: {y_train.sum()}")
print(f"Tamanho do teste: {X_test.shape[0]} | Fraudes no teste: {y_test.sum()}")

# ------------------------------------------------------------------------------
# 4. Treinamento e Comparação de Modelos
# ------------------------------------------------------------------------------
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"Treinando {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'Recall': rec,
        'Precision': prec,
        'F1-Score': f1
    }

metrics_df = pd.DataFrame(results).T[['Recall', 'Precision', 'F1-Score']]
print("\n=== Comparação dos Modelos (Limiar Padrão = 0.5) ===")
print(metrics_df)

# Experimento SMOTE com Random Forest
print("\n=== Treinando Random Forest + SMOTE ===")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_res, y_train_res)

y_pred_smote = rf_smote.predict(X_test)
print("\nRelatório Random Forest + SMOTE:")
print(classification_report(y_test, y_pred_smote, target_names=['Normal', 'Fraude']))

# ------------------------------------------------------------------------------
# 5. Avaliação e Ajuste de Limiar (Threshold Tuning)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_proba'])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})')

plt.xlabel('Recall (Sensibilidade)')
plt.ylabel('Precision (Precisão)')
plt.title('Curva Precision-Recall por Modelo', fontsize=14)
plt.legend()
plt.show()

# Otimização de Limiar no XGBoost
y_proba_xgb = results['XGBoost']['y_proba']
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_xgb)

# threshold array tem tamanho N, precisions e recalls tem tamanho N + 1
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nMelhor Limiar de Decisão para XGBoost: {best_threshold:.4f}")
print(f"Recall: {recalls[best_idx]:.4f} | Precisão: {precisions[best_idx]:.4f} | F1-Score: {f1_scores[best_idx]:.4f}")

y_pred_adj = (y_proba_xgb >= best_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_adj)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'])
plt.title(f'Matriz de Confusão XGBoost (Limiar={best_threshold:.3f})')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

# ------------------------------------------------------------------------------
# 6. Explicabilidade do Modelo com SHAP
# ------------------------------------------------------------------------------
print("\nGerando explicações do SHAP...")
explainer = shap.TreeExplainer(results['XGBoost']['model'])
shap_values = explainer.shap_values(X_test)

# Resumo em Barras
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Beeswarm Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)
