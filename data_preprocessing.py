# Data Preprocessing

# Importer les librairies
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Importer le dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#%%
# Gérer les données manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(X[:, 1:3])
X[:,1:3]= imputer.transform(X[:, 1:3])
#%%
# Gerer les variables categoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X
#%%
# enlever relation relation d'ordre en creant colomnnes
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X
#%%
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#%%
# features scaling mettre a l'echelle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # creer un objet de la class standardScaler()
X_train = sc.fit_transform(X_train) # lier objet ala matrice de fatures du training set et la sclaer
X_test = sc.transform(X_test)
X_test
#%%
# REGRESSION LINEAIRE SIMPLE
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
#%%
# REGRESSION LINEAIRE MULTIPLE implementation
# backward
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Importer le dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%%
# Gerer les variables categoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])


#%%
# enlever relation relation d'ordre en creant colomnnes
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # enlever 1 colonne, piege des dummy variables
#%%
# Diviser en training et test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
# construction du modele multiple
#%%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#%%
#  Faire de nouvelles predictions
y_pred  = regressor.predict(X_test) # valeur des VI dont on peut predire la valeur des VD
y_pred # predictions a partir de valeurs dans le test set que nous avonc construit
#%%
regressor.predict(np.array([[1, 0, 130000, 140000, 300000]]))# predictions vous VIqui ne sont pas dans le X_dataset
#%%
# Linear regression polynomiale
#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Importer le dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values
#%%
# construction du modele non lineaire
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)
#%%

# visualiser les resultats
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X_poly), color = 'blue')
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()
#%%
