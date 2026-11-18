import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
#chargement de base de données iris
iris = datasets.load_iris()
# choix de deux variables : Utiliser les deux premières colonnes afin d'avoir un problème de classification binaire
X = iris.data[:, :2]
y = (iris.target != 0) * 1 # re-étiquetage des fleurs
# import matplotlib.pyplot as plt

# plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Setosa (0)')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Versicolor/Virginica (1)')
# plt.xlabel('Sepal length (cm)')
# plt.ylabel('Sepal width (cm)')
# plt.title('Iris : longueur et largeur des sépales')
# plt.legend()
# plt.grid(True)
# plt.show()

## TRAIN
model = LogisticRegression(C=1e20)
model.fit(X, y)
# TEST : les quatre valeurs demandées
tests = [[5.5, 2.5], [7, 3], [3, 2], [5, 3]]
for point in tests:
    classe = model.predict([point])[0]
    proba = model.predict_proba([point])[0]
    print(f"{point} -> classe {classe} | proba setosa = {proba[0]:.3f}, proba autre = {proba[1]:.3f}")