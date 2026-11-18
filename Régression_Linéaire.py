import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv("C:\\Users\\MSI\\Downloads\\Atelier 3\\Atelier 3\\DB\\DB\\univariateLinearRegression.csv")
print(len(df))
#selection de la première colonne de la base (la taille de la population)
X = df.iloc[0:len(df),0]
#selection de la deuxième colonne de la base (le profit effectué)
Y = df.iloc[0:len(df),1]
plt.scatter(X, Y, c='blue', alpha=0.7)
plt.xlabel('Population de la ville (en 10 000 habitants)')
plt.ylabel('Profit (en 10 000 $)')
plt.title('Profit en fonction de la population')
plt.grid(True)
plt.show()
#linregress() renvoie plusieurs variables de retour. On s'interesse au slope et intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
def F(x):
  return slope * x + intercept
#la variable fitLine est un tableau de valeurs prédites depuis la tableau de variables X
fitLine = F(X)
plt.scatter(X,Y)
plt.plot(X, fitLine, c='r')
plt.show() 