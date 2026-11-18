
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df_sal = pd.read_csv("C:\\Users\\MSI\\Downloads\\Atelier 3\\Atelier 3\\DB\\DB\\Position_Salaries.csv")
df_sal.head()
# # a) Description de la base
# print(df_sal.head())
# print(df_sal.describe())
# print(df_sal.info())

# # b) Distribution des données avec Seaborn
# sns.histplot(df_sal['Salary'], kde=True)
# plt.title('Distribution des salaires')
# plt.show()

# # b) Scatter plot : salaire en fonction du niveau
# plt.scatter(df_sal['Level'], df_sal['Salary'], c='blue')
# plt.xlabel('Niveau')
# plt.ylabel('Salaire')
# plt.title('Salaire en fonction du niveau')
# plt.grid(True)
# plt.show()

X = df_sal.iloc[:, 1:-1].values # independent (level=experience )
y = df_sal.iloc[:, -1].values # dependent (salary)
# Train linear regression model
lr = LinearRegression()
lr.fit(X, y)
# Train polynomial regression model
pr = PolynomialFeatures(degree = 4)
X_poly = pr.fit_transform(X)
lr_2 = LinearRegression()
lr_2.fit(X_poly, y)

# TEST
y_pred_lr = lr.predict(X) # Linear Regression
y_pred_poly = lr_2.predict(X_poly) # Polynomial Regression
# a) Graphique : données réelles et prédictions des deux modèles
X_grille = np.arange(X.min(), X.max() + 0.1, 0.1).reshape(-1, 1)

plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X_grille, lr.predict(X_grille), color='green', label='Régression linéaire')
plt.plot(X_grille, lr_2.predict(pr.transform(X_grille)), color='red', label='Régression polynomiale (degré 4)')
plt.xlabel('Niveau')
plt.ylabel('Salaire')
plt.title('Salaire en fonction du niveau')
plt.legend()
plt.grid(True)
plt.show()

# b) Prédiction pour un niveau de 7,5
print(f"Régression linéaire : {lr.predict([[7.5]])[0]:,.0f}")
print(f"Régression polynomiale : {lr_2.predict(pr.transform([[7.5]]))[0]:,.0f}")