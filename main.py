import os
import tiktoken
from tokenizers import ByteLevelBPETokenizer
from flask import Flask, render_template, request

app = Flask(__name__)
tokenizer = ByteLevelBPETokenizer()


with open('textos/texto', 'r', encoding='utf-8') as file:
    dados = file.read()
    
PATH = 'textos/texto'

# Initialize a ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
# Treinar o tokenizer
tokenizer.train(files=[PATH], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
]) 
    
@app.route('/')
def home():
    return render_template('home.html')




@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    print(tokenizer.encode(query).ids)
    return render_template('search.html', query=query)





if __name__ == '__main__': 
    os.environ['FLASK_ENV'] = 'development'  
    app.run(debug=True)