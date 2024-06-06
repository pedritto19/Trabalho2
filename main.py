import os
import numpy as np
import base64
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Carregar o modelo treinado
model = load_model('trained_model.h5')

# Definir as dimensões da imagem
img_height, img_width = 150, 150

# Definir o mapeamento das classes
class_indices = {'cavalo': 0, 'passarinho': 1}

def predict_image(model, image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction

def get_class_label(prediction, class_indices):
    class_label = None
    max_prob = np.max(prediction)
    for label, index in class_indices.items():
        if prediction[0][index] == max_prob:
            class_label = label
            break
    return class_label

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['POST'])
def search():
    if 'query' not in request.files:
        return "No file part"
    
    file = request.files['query']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Salvar a imagem carregada
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Fazer a predição
        prediction = predict_image(model, file_path, img_height, img_width)
        
        # Obter o rótulo da classe
        class_label = get_class_label(prediction, class_indices)
        
        # Exibir o resultado
        resultado = f"A imagem é de um(a): {class_label}"
        
        # Converter a imagem para base64
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Deletar a imagem após a conversão
        os.remove(file_path)
        
        # Gerar o prefixo data URI para a imagem
        image_data = f"data:image/jpeg;base64,{encoded_string}"
        
        return render_template('search.html', query=image_data, resultado=resultado)

if __name__ == '__main__': 
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    os.environ['FLASK_ENV'] = 'development'  
    app.run(debug=True)