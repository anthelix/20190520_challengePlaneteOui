import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values     # variables independantes
y = dataset.iloc[:, -1].values      #variables dependantes

# pas de donnes manquantes dans ce dataset pas de variable scategoriques mais numerique continue
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1.0/3, random_state = 0)
# pas de Feature scaling dans la regression lineaire car tout sera a la meme echelle du fait u coef

# Construction du modele
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#  Faire de nouvelles predictions
X_test
y_pred  = regressor.predict(X_test) # valeur des VI dont on peut predire la valeur des VD
y_pred # predictions a partir de valeurs dans le test set que nous avonc construit
regressor.predict([[15]])# predictions vous VIqui ne sont pas dans le X_dataset

# Visulaliser les resultats
plt.scatter(X_test, y_test, color = 'red')   # placer des points
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # tracer une droite, ordonnes regressor car on veut la predictio
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()