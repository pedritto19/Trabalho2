import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)




    
@app.route('/')
def home():
    return render_template('home.html')




@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']


    return render_template('search.html', query=query)





if __name__ == '__main__': 
    os.environ['FLASK_ENV'] = 'development'  
    app.run(debug=True)