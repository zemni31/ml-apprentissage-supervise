import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# 1. Chargement : la première ligne du fichier sert d'en-tête
diabetes_data = pd.read_csv(
    "C:\\Users\\MSI\\Downloads\\Atelier 3\\Atelier 3\\DB\\DB\\diabetes.csv"
)
print(diabetes_data.columns.tolist())   # vérifier les noms réels des colonnes

# 2. Séparation des variables : X = caractéristiques, y = cible
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_data[feature_cols]
y = diabetes_data['Outcome']

# 3. Découpage apprentissage / test (question 10a)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# 4. Arbre de décision avec l'indice de Gini (question 10b)
clf = DecisionTreeClassifier(criterion='gini', random_state=1)
clf = clf.fit(X_train, y_train)

# 5. Prédiction sur le jeu de test
y_pred = clf.predict(X_test)

# 6. Évaluation (question 11)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)