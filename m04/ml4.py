import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, accuracy_score

data_train = pd.read_csv("pendigits.tra", header=None)
data_test = pd.read_csv("pendigits.tes", header=None)

x_train_full, y_train_full = data_train.iloc[:, :-1], data_train.iloc[:, -1]
x_test, y_test = data_train.iloc[:, :-1], data_train.iloc[:, -1]

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
perceptron.fit(x_train, y_train)

print("Perceptron:")
print("  Accuracy train:", accuracy_score(y_train, perceptron.predict(x_train)))
print("  Accuracy val  :", accuracy_score(y_val, perceptron.predict(x_val)))
print("  Accuracy test :", accuracy_score(y_test, perceptron.predict(x_test)))

mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=300, random_state=42)
mlp.fit(x_train, y_train)

print("\nMLPClassifier:")
print("  Accuracy train:", accuracy_score(y_train, mlp.predict(x_train)))
print("  Accuracy val  :", accuracy_score(y_val, mlp.predict(x_val)))
print("  Accuracy test :", accuracy_score(y_test, mlp.predict(x_test)))

alphas = [0.0001, 0.001, 0.01, 0.1]
lrs = [0.0001, 0.001, 0.01]
results = {}

for lr in  lrs:
    train_scores, val_scores = [], []
    for a in alphas:
        model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=a, learning_rate_init=lr, max_iter=300, random_state=42)
        model.fit(x_train, y_train)
        train_scores.append(accuracy_score(y_train, model.predict(x_train)))
        val_scores.append(accuracy_score(y_val, model.predict(x_val)))

    results[lr] = (train_scores, val_scores)



for lr in lrs:
    train_scores, val_scores = results[lr]
    plt.plot(alphas, train_scores, label=f"Train (lr={lr}")
    plt.plot(alphas, val_scores, label=f"Val (lr={lr}", linestyle="--")


plt.xscale("log")
plt.xlabel("Alpha (регуляризация)")
plt.ylabel("Accuracy")
plt.title("Влияние alpha и learning_rate на точность MLP")
plt.legend()
plt.show()


