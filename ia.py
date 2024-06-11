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




































import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Definindo os parâmetros
img_width, img_height = 150, 150
batch_size = 32
epochs = 50
num_classes = 8  # Número de classes (raças de cachorro)

# Caminhos para os dados de treino e validação
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# Pré-processamento das imagens
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Construindo o modelo
model = Sequential([
    tf.keras.Input(shape=(img_width, img_height, 3)),
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

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint para salvar o melhor modelo
checkpoint = ModelCheckpoint('dog_breed_classifier.keras', monitor='val_loss', save_best_only=True, mode='min')

# Treinando o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint]
)

# Salvando o modelo final
model.save('dog_breed_classifier_final.keras')

print("Modelo treinado e salvo com sucesso!")