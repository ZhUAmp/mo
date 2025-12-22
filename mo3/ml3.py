import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


data = pd.read_csv("sml2010.txt", sep=r"\s+")



data = data.drop(columns=["#", "1:Date", "2:Time"])

y = data["3:Temperature_Comedor_Sensor"]

X = data.drop(columns=["3:Temperature_Comedor_Sensor"])

X = X.dropna(axis=1, how='all')  #удаляем колонки, где все значения NaN
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

train_size = int(0.7 * len(X))
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#лин

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_train = lin_reg.predict(X_train)
y_pred_test = lin_reg.predict(X_test)

print("Линейная:")
print("R2 train:", r2_score(y_train, y_pred_train))
print("R2 test:", r2_score(y_test, y_pred_test))




#Полиномиальная регрессия
degrees = [1, 2, 3, 4, 5]
train_scores, test_scores = [], []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_scores.append(r2_score(y_train, model.predict(X_train_poly)))
    test_scores.append(r2_score(y_test, model.predict(X_test_poly)))

plt.plot(degrees, train_scores, label="Train R2")
plt.plot(degrees, test_scores, label="Test R2")
plt.xlabel("Степень полинома")
plt.ylabel("R2")
plt.title("Полиномиальная регрессия")
plt.legend()
plt.show()

#Ridge-регрессия
alphas = [0.01, 0.1, 1, 10, 100]
train_scores, test_scores = [], []

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    train_scores.append(r2_score(y_train, model.predict(X_train)))
    test_scores.append(r2_score(y_test, model.predict(X_test)))

plt.plot(alphas, train_scores, label="Train R2")
plt.plot(alphas, test_scores, label="Test R2")
plt.xscale("log")
plt.xlabel("Коэффициент регуляризации (alpha)")
plt.ylabel("R2")
plt.title("Ridge-регрессия")
plt.legend()
plt.show()









