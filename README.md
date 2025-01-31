# Maschinelles Lernen zur Früherkennung von Diabetes

## 🔍 Ziel des Projekts: Diabetes frühzeitig erkennen

Dieses Projekt beschäftigt sich mit der Vorhersage von Diabetes mithilfe von maschinellem Lernen. Es basiert auf einem öffentlichen Kaggle-Datensatz, der medizinische Messwerte von Patienten enthält. Ziel ist es, verschiedene Machine-Learning-Modelle zu vergleichen, um das beste Modell zur Vorhersage von Diabetes zu finden. Die Modelle **XGBoost**, **Random Forest** und ein **Ensemble-Modell** werden trainiert und optimiert, um die besten Ergebnisse zu erzielen.

## 📂 Datensatz: Medizinische Gesundheitsdaten

Der Datensatz stammt aus [Kaggle](https://www.kaggle.com/datasets/nancyalaswad90/review) und enthält wichtige Gesundheitsmetriken von Patienten. Diese umfassen:

- **Glucose**: Blutzuckerwert
- **BMI (Body Mass Index)**: Maß für das Körpergewicht in Relation zur Körpergröße
- **Blood Pressure (Blutdruck)**: Wichtige Metrik zur Beurteilung des Gesundheitszustands
- **Pregnancies (Schwangerschaften)**: Anzahl der Schwangerschaften
- **DiabetesPedigreeFunction**: Eine Metrik, die das Risiko für Diabetes basierend auf der genetischen Veranlagung bewertet
- **Age (Alter)**: Das Alter des Patienten

Das Ziel ist es, basierend auf diesen Features vorherzusagen, ob eine Person an Diabetes erkrankt ist oder nicht (Zielvariable `Outcome`).

##  Auswahl der Modelle: Warum XGBoost & Random Forest?

- **XGBoost**: Ein extrem leistungsstarker Algorithmus, der besonders gut mit strukturierten Daten arbeitet und sich für Klassifikationsaufgaben eignet.
- **Random Forest**: Ein Ensemble-Algorithmus, der aus mehreren Entscheidungsbäumen besteht. Er ist robust gegenüber Overfitting und bietet gute Interpretierbarkeit.
- **Voting Classifier (Ensemble-Modell)**: Kombination aus XGBoost und Random Forest zur Erhöhung der Gesamtgenauigkeit.

##  Datenaufbereitung: Feature-Engineering & Standardisierung

- **Feature-Engineering**: Eine neue Spalte `Glucose_BMI` wird erstellt, um mögliche Wechselwirkungen zwischen Blutzucker und BMI zu erfassen.
- **Standardisierung**: Die Werte werden skaliert, um eine bessere Performance der Modelle zu gewährleisten.

```python
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## 📌 Wichtige Begriffe & Methoden

- **GridSearchCV**: Eine Technik zur Hyperparameter-Optimierung, die verschiedene Kombinationen von Parametern testet, um das beste Modell zu finden.
- **Precision (Präzision)**: Anteil der korrekt vorhergesagten positiven Fälle.
- **Recall (Sensitivität)**: Anteil der tatsächlich positiven Fälle, die korrekt erkannt wurden.
- **F1-Score**: Ein Maß, das Precision und Recall kombiniert.

##  Modelltraining: Optimierung & Hyperparameter

### 1️⃣ XGBoost mit Hyperparameter-Tuning

```python
xgb_params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'gamma': [0, 0.1, 0.2],
    'scale_pos_weight': [len(y_train) / sum(y_train)]
}
```

### 2️⃣ Random Forest mit optimierten Parametern

```python
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'class_weight': ['balanced', 'balanced_subsample']
}
```

### 3️⃣ Ensemble-Modell für höhere Genauigkeit

```python
ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
```

## 📊 Auswertung: Klassifikationsberichte der Modelle

###  Klassifikationsbericht für XGBoost:

- **Genauigkeit (Accuracy):** 70%
- **Präzision (Precision):**
  - Klasse 0: 89%
  - Klasse 1: 55%
- **Sensitivität (Recall):**
  - Klasse 0: 62%
  - Klasse 1: 85%
- **F1-Score:**
  - Klasse 0: 73%
  - Klasse 1: 67%

###  Klassifikationsbericht für Random Forest:

- **Genauigkeit (Accuracy):** 73%
- **Präzision (Precision):**
  - Klasse 0: 81%
  - Klasse 1: 61%
- **Sensitivität (Recall):**
  - Klasse 0: 77%
  - Klasse 1: 67%
- **F1-Score:**
  - Klasse 0: 79%
  - Klasse 1: 64%

###  Klassifikationsbericht für Ensemble-Modell:

- **Genauigkeit (Accuracy):** 75%
- **Präzision (Precision):**
  - Klasse 0: 86%
  - Klasse 1: 61%
- **Sensitivität (Recall):**
  - Klasse 0: 73%
  - Klasse 1: 78%
- **F1-Score:**
  - Klasse 0: 79%
  - Klasse 1: 68%

## 📈 Einfluss der Features auf die Modellentscheidungen

### Feature-Wichtigkeit im XGBoost-Modell:

Die folgende Grafik zeigt die wichtigsten Merkmale für das XGBoost-Modell. Auffällig ist, dass `Glucose_BMI` die höchste Bedeutung hat. Dies bestätigt, dass die Kombination aus Blutzuckerwerten und BMI eine starke Korrelation mit Diabetes hat.

![Figure_1](https://github.com/user-attachments/assets/b712d345-7a8f-4720-986f-9fe723f8f664)


### Feature-Wichtigkeit im Random Forest-Modell:

Im Random Forest-Modell haben `Glucose_BMI` und `Glucose` die höchsten Einflusswerte. Das bestätigt die Bedeutung von Blutzuckerwerten für die Diabetes-Vorhersage. Auch das **Alter** (`Age`) und der **BMI** spielen eine wesentliche Rolle.

![Figure_2](https://github.com/user-attachments/assets/dac8132c-99bc-401c-8cbb-0aeb72ffda62)


## 📌 Erkenntnisse & Optimierungspotenzial

- **XGBoost liefert die beste Leistung**, insbesondere für die Identifikation von Diabetes-Fällen.
- **Das Ensemble-Modell erhöht die Stabilität**, indem es mehrere Vorhersagen kombiniert.
- **Feature-Engineering ist entscheidend**, da `Glucose_BMI` als neues Feature signifikant zur Vorhersage beigetragen hat.
- **GridSearchCV hat geholfen, die besten Parameter zu finden**, was die Modellleistung erheblich verbessert hat.


---
🖥️ Nutzung der Streamlit-App

Gib die Gesundheitswerte ein: Benutzer können verschiedene Parameter eingeben.

Klicke auf "Vorhersage starten": Das Modell berechnet das Diabetes-Risiko.

Ergebnisse anzeigen: Die App zeigt an, ob Diabetes festgestellt wurde oder nicht.

<img width="645" alt="Bildschirmfoto 2025-01-31 um 13 56 50" src="https://github.com/user-attachments/assets/9e34ecff-145d-4c04-95d5-d4cc9bedd359" />

---
Falls du Fragen hast oder Verbesserungen vorschlagen möchtest, erstelle gerne ein Issue oder einen Pull-Request! 😊

