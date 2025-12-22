import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image


model = keras.models.load_model("cnn_model.h5")
print("Модель загружена.")


def load_image(path):

    img = Image.open(path).convert("L")
    img = img.resize((28, 28))
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img



try:
    path = input("Введите путь к изображению цифры (PNG/JPG): ")
    img = load_image(path)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    print(f"Предсказанная цифра: {digit}")

except Exception as e:
    print("Ошибка:", e)
    print("Можете также проверить модель на встроенном тестовом наборе.")

