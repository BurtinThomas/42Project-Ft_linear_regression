from sklearn.linear_model import LinearRegression
import pandas
import numpy as np

df = pandas.read_csv('data.csv')
x = df['km'].values
y = df['price'].values
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

model = LinearRegression()
model.fit(x, y)

print(f"Pente (coefficient) : {model.coef_[0]}")
print(f"Ordonnée à l'origine : {model.intercept_}")

kilometrage_input = np.array([[240000]])
predict_prices = model.predict(kilometrage_input)
print(f"prix prévue pour {kilometrage_input[0, 0]}km : {predict_prices[0, 0]}$")