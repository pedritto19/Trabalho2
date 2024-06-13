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
model = load_model('dog_breed_classifier_final.keras')

# Definir as dimensões da imagem
img_height, img_width = 150, 150

# Definir o mapeamento das classes
class_indices = {
    'Pastor-alemão': 0,
    'Buldogue': 1,
    'Labrador retriever': 2,
    'Golden retriever': 3,
    'Buldogue francês': 4,
    'Beagle': 5,
    'Husky siberiano': 6,
    'Poodle': 7
}


# Lista de raças e URLs de imagens
dog_breeds = [
    ('Pastor-alemão', 'https://www.doglife.com.br/blog/assets/post/pastor-alemao-guia-completo-sobre-a-raca-62e7ecf756767a00fc5ebc9a/pastor-alemao-guia.jpg'),
    ('Buldogue', 'https://www.blog.dogtripbrasil.com.br/wp-content/uploads/2020/03/BULDOGUE-INGL%C3%8AS.jpg'),
    ('Labrador retriever', 'https://p2.trrsf.com/image/fget/cf/940/0/images.terra.com/2023/11/29/1147115027-labrador-retriever.jpg'),
    ('Golden retriever', 'https://www.petz.com.br/blog/wp-content/uploads/2017/06/golden-retriever.jpg'),
    ('Buldogue francês', 'https://panoramapetvet.com.br/wp-content/uploads/2023/08/Design-sem-nome-100.jpg'),
    ('Husky siberiano', 'https://www.petz.com.br/cachorro/racas/husky-siberiano/img/husky-siberiano-caracteristicas-guia-racas.jpg'),
    ('Beagle', 'https://diariodonordeste.verdesmares.com.br/image/contentid/policy:1.3136324:1631737586/beagle-ra%C3%A7a.jpg?f=16x9&$p$f=44393c1'),
    ('Malamute-do-alasca', 'https://petnovitta.com.br/wordpress/wp-content/files/petnovitta.com.br/2023/09/malamute-768x512.png')
]


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
    return render_template('home.html', dog_breeds=dog_breeds)

@app.route('/search', methods=['POST'])
def search():
    if 'query' not in request.files:
        return render_template('home.html', dog_breeds=dog_breeds, error="Nenhuma imagem foi selecionada.")
    
    file = request.files['query']
    
    if file.filename == '':
        return render_template('home.html', dog_breeds=dog_breeds, error="Nenhuma imagem foi selecionada.")
        # Verificar se o arquivo é uma imagem válida
    if not file.content_type.startswith('image/'):
        return render_template('home.html', dog_breeds=dog_breeds, error="Selecione uma imagem válida.")
    
    if file:
        # Salvar a imagem carregada
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Fazer a predição
        prediction = predict_image(model, file_path, img_height, img_width)
        
        # Obter o rótulo da classe
        class_label = get_class_label(prediction, class_indices)
        
        # Exibir o resultado
        resultado = f"Seu cachorro é da raça {class_label} 💖"
        
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