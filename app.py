from flask import Flask, render_template
from docplex.mp.model import Model
import numpy as np
import random
from flask import Flask, render_template, request,jsonify
import csv
import os
import json

app = Flask(__name__)
app.debug = True  # Enables auto-reload

@app.route('/')
def home():
    return render_template('index.html', show_input_form=True)

@app.route('/cplex_solver')
def cplex_solver():
    return render_template('cplex_solver.html', show_input_form=True)

@app.route('/genetic_solver')
def genetic_solver():
    return render_template('genetic_solver.html', show_input_form=True)





@app.route('/solve_genetic_algorithm', methods=['POST'])
def solve_genetic_algorithm():
    try:
            if 'data_file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
                
            file = request.files['data_file']

            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # Vérifier que le fichier est un fichier JSON
            if file and file.filename.endswith('.json'):
                # Lire directement le fichier en mémoire
                data = json.load(file)         
                 # Accessing and printing each data part
                print("Population Size:", data.get('population_size'))
                print("Number of Generations:", data.get('num_generations'))
            # Réponse avec les données traitées
            return jsonify({'message': 'File processed successfully!', 'data': data}), 200
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format'}), 400
       


if __name__ == '__main__':
    app.run(debug=True)