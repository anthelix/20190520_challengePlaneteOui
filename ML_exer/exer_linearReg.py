import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

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





X1 = dataset.iloc[:, :-1].values
y1 = dataset.iloc[:, -1].values
#visualize dataset
plt.plot(X1 ,y1, 'o')
plt.xlabel('Experience')
plt.ylabel('Salaire')



def forward_propagation(train_dataset, parameters):
    w = parameters['w']
    b = parameters['b']
    predictions = np.multiply(w, X_train) + b
    return predictions

def cost_function(predictions, y_train):
    cost = np.mean((y_train - predictions)**2)*0.5
    return cost

def backward_propagation(X_train, y_train, predictions):
    derivatives = dict() # function creates a dictionary
    df = (y_train - predictions) * -1  # dc/df = (y-f)*-1
    dw = np.mean(np.multiply(X_train, df))
    db = np.mean(df)
    derivatives['dw'] = dw
    derivatives['db'] = db
    return derivatives

def update_parameters(parameters, derivatives, learning_rate):
    parameters['w'] = parameters['w'] - learning_rate * derivatives['dw']
    parameters['b'] = parameters['b'] - learning_rate * derivatives['db']
    return parameters

def train(X_train, y_train, learning_rate=, iters = 10):
    # parametres aleatoires
    parameters = dict()
    parameters['w'] = np.random.uniform(0.1) * -1
    parameters['b'] = np.random.uniform(0.1) * -1
    plt.figure()
    time.sleep(6) 
    # loss
    loss = list()
    # iterate
    for i in range(iters):
        #forward propagation
        predictions = forward_propagation(X_train, parameters)

        #cost function
        cost = cost_function(predictions, y_train)

        #append loss
        loss.append(cost)

        print("Iteration = {}, Loss = {}".format(i+1, cost))
        time.sleep(6)
        #plot
        plt.plot(X_train, y_train, 'o')
        plt.plot(X_train, predictions, 'x')
        plt.show()
        time.sleep(6)
        #back probagation
        derivatives = backward_propagation(X_train, y_train, predictions)

        #update parameters
        parameters = update_parameters(parameters, derivatives, learning_rate)

    return parameters, loss

parameters, loss = train(X_train, y_train, 0.01, 20)


pred = X_test * parameters['w'] + parameters['b']
plt.figure()
plt.plot(X_test, y_test, 'o')
plt.plot(X_test, pred, 'x', color='pink')
plt.show()

