"""
@author: Loglisci Raffaele

Questo modulo analizza un dataset riguardante gliomi correlati a mutazioni molecolari
utilizzando una RETE NEURALE. Viene effettuato il caricamento dei dati, la loro elaborazione,
l'addestramento del modello, e l'analisi delle sue prestazioni attraverso metriche e grafici.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, \
    average_precision_score, precision_recall_curve, f1_score, accuracy_score
from inspect import signature

# caricamento il dataset
data = pd.read_csv('../2.Ontologia/TCGA_InfoWithGrade.csv')

# esplorazione del dataset
print(data.info())

# divido i dati in features di input e feature target
y = data['Grade']
X = data.drop('Grade', axis=1)

# divido il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)


# Costruisco il modello della rete neurale
def create_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compilazione del modello
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# creazione istanza model1 e addestramento
model1 = create_model()
model1.fit(X_train, y_train, epochs=30, batch_size=64)

# valutazione delle prestazioni del modello
test_loss, test_accuracy = model1.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# calcolo delle previsioni e arrotondamento
predictions = model1.predict(X_test)
rounded = [round(x[0]) for x in predictions]

# stampa del report di classificazione e della matrice di confusione
print('\nClasification report:\n', classification_report(y_test, rounded))
print('\nConfusion matrix:\n', confusion_matrix(y_test, rounded))

# visualizzazione della matrice di confusione tramite heatmap
confusion_matrix_values = confusion_matrix(y_test, rounded)
df_cm = pd.DataFrame(confusion_matrix_values, index=[i for i in "01"], columns=[i for i in "01"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.show()

# k-fold Cross-Validation (manual implementation)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kfold.split(X_train):
    # Definire i set di addestramento e validazione
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Crea e addestra il modello
    model = create_model()
    model.fit(X_train_fold, y_train_fold, epochs=30, batch_size=64, verbose=0)

    # Calcola le previsioni e valuta l'accuratezza
    y_val_pred = np.round(model.predict(X_val_fold))
    acc = accuracy_score(y_val_fold, y_val_pred)
    cv_scores.append(acc)

# Stampa delle statistiche della cross-validation
print('\nCV Scores:', cv_scores)
print('CV Scores Mean:', np.mean(cv_scores))
print('CV Scores Variance:', np.var(cv_scores))
print('CV Scores Standard Deviation:', np.std(cv_scores))

# Calcolo del AUC per la curva di ROC
probs = model1.predict(X_test)[:, 0]
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# calcola roc curve e visualizzazione grafica
fpr, tpr, thresholds = roc_curve(y_test, probs)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
pyplot.show()

# Calcolo della curva Precision-Recall e visualizzazione
average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)

# in matplotlib versione < 1.5, plt.fill_between non dispone dell'argomento 'step'.
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# Calcolo e visualizzazione dell'F1-score
f1 = f1_score(y_test, rounded)
print('\nf1 score: ', f1)

# Creazione di un grafico per visualizzare varianza e deviazione standard dei cv_scores
data = {'Variance': np.var(cv_scores), 'Standard Deviation': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()
