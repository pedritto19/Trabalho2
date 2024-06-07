# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
# from tensorflow.keras.models import load_model

# def predict_image(model, image_path, img_height, img_width):
#     img = load_img(image_path, target_size=(img_height, img_width))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     prediction = model.predict(img_array)
#     return prediction

# def get_class_label(prediction, class_indices):
#     class_label = None
#     max_prob = np.max(prediction)
#     for label, index in class_indices.items():
#         if prediction[0][index] == max_prob:
#             class_label = label
#             break
#     return class_label

# # Carregar o modelo treinado
# model = load_model('trained_model.h5')

# # Definir as dimensões da imagem
# img_height, img_width = 150, 150

# # Definir o mapeamento das classes
# class_indices = {'cavalo': 0, 'passarinho': 1}

# # Caminho da imagem a ser classificada
# image_path = 'teste.jpg'

# # Fazer a predição
# prediction = predict_image(model, image_path, img_height, img_width)

# # Obter o rótulo da classe
# class_label = get_class_label(prediction, class_indices)

# # Exibir o resultado
# print(f"A imagem é de um(a): {class_label}")




































import os
import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Certifique-se de que a biblioteca Pillow está instalada
try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    from PIL import Image

# Configurações
data_dir = './data'
img_height, img_width = 150, 150
batch_size = 32
num_classes = 2
epochs = 25

# Geradores de dados para treinamento e validação
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  # 20% dos dados serão usados para validação
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Construção do modelo
model = Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilação do modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Salvando o modelo treinado
model.save('trained_model.h5')

# Avaliação do modelo
loss, accuracy = model.evaluate(validation_generator)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')