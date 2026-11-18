import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
df = pd.read_excel("C:\\Users\\MSI\\Downloads\\Atelier 3\\Atelier 3\\DB\\DB\\multivariateLinearRegression.xlsx")
print(df.head())
#Récupérer le prix : les valeurs observées pour la variable Cible
Y = df["prix"]
#Récupérer les variables prédictives : La superficie en pieds² et le nb chambre
X = df[['taille_en_pieds_carre','nb_chambres']]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df['taille_en_pieds_carre'], Y, c='blue', alpha=0.7)
axes[0].set_xlabel('Taille (pieds²)')
axes[0].set_ylabel('Prix de vente ($)')
axes[0].set_title('Prix en fonction de la taille')
axes[0].grid(True)

axes[1].scatter(df['nb_chambres'], Y, c='green', alpha=0.7)
axes[1].set_xlabel('Nombre de chambres')
axes[1].set_ylabel('Prix de vente ($)')
axes[1].set_title('Prix en fonction du nombre de chambres')
axes[1].grid(True)

plt.tight_layout()
plt.show()
scale = StandardScaler()
X_scaled = scale.fit_transform(X)
print(X_scaled)
model_brut = sm.OLS(Y, sm.add_constant(X)).fit()
print(model_brut.summary())
def predict_price_of_house(taille_maison, nb_chambre):
    p = model_brut.params
    return p['const'] + p['taille_en_pieds_carre'] * taille_maison + p['nb_chambres'] * nb_chambre
print(predict_price_of_house(4500,5))