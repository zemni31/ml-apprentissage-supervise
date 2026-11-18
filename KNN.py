import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import *
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
digit = load_digits()
dig = pd.DataFrame(digit['data'][0:1700])
print(dig.head())
print(digit.keys())
#pour voir la 5ème image du jeu de données
#plt.imshow(digit['images'][4], cmap='Greys_r')
#plt.show()

train_x = digit.data # input variables
train_y = digit.target # les étiquettes (output variable)
#découpage du jeu de données (0.25 pour indiquer 25%)
x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=
0.25)

### Train
#KNN = KNeighborsClassifier(7) # on utilise 7 voisins
#KNN.fit(x_train, y_train)

###Test
#print(KNN.score(x_test,y_test))

# 3. Validation croisée à 5 plis sur le jeu d'apprentissage uniquement
k_values = list(range(1, 21))
mean_scores = []

start = time.time()
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5)  # 5 essais
    mean_scores.append(scores.mean())
    print(f"K = {k:2d} | scores des 5 plis : {np.round(scores, 4)} | moyenne = {scores.mean():.4f}")
print(f"Temps de la validation croisée : {time.time() - start:.2f} s")

# 4. Choix du meilleur K selon la moyenne des 5 essais
best_k = k_values[int(np.argmax(mean_scores))]
print(f"\nMeilleur K : {best_k} (moyenne = {max(mean_scores):.4f})")

# 5. Réentraînement avec le meilleur K sur tout le jeu d'apprentissage
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(x_train, y_train)

# 6. Test final, réalisé une seule fois
print(f"Score final sur le jeu de test : {best_knn.score(x_test, y_test):.4f}")