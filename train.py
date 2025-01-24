import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Utilisation de MinMaxScaler

def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta

def main():
    try:
        df = pd.read_csv('data.csv')
        x = df['km'].values
        y = df['price'].values
        x = x.reshape(x.shape[0], 1)
        y = y.reshape(y.shape[0], 1)

        # Utilisation de MinMaxScaler pour la normalisation (de 0 à 1)
        scaler = MinMaxScaler()
        x_norm = scaler.fit_transform(x)

        billet = np.ones(x_norm.shape)
        X = np.concatenate((x_norm, billet), axis=1)

        theta = np.array([0, 0]).reshape(2, 1)
        theta_final = gradient_descent(X, y, theta, 0.1, 7000)

        theta_final[0] = theta_final[0] / (scaler.data_max_[0] - scaler.data_min_[0])
        theta_final[1] = theta_final[1] - theta_final[0] * scaler.data_min_[0]
        XX = np.concatenate((x, billet), axis=1)
        y_pred = model(XX, theta_final)
        print(theta_final)
        print(theta_final[0] * 240000 + theta_final[1])

        plt.scatter(x, y, label='Données réelles')
        plt.plot(x, y_pred, color='red', label='Régression linéaire')
        plt.title("Régression Linéaire")
        plt.xlabel("Kilomètres")
        plt.ylabel("Prix")
        plt.legend()
        plt.show()
    except KeyboardInterrupt:
        return
    except Exception as error:
        print("Error: {error}")

main()
