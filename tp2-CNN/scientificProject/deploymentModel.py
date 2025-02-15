from flask import Flask, request, jsonify
from flask_cors import CORS  # Importer CORS
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Charger le modèle
model = tf.keras.models.load_model("fruits_model.h5")

# Classes des fruits
class_names = ['apple', 'banana', 'orange']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Vérifiez si un fichier est inclus dans la requête
        if 'file' not in request.files:
            return jsonify({'error': 'Aucune image reçue'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Le fichier est vide'}), 400

        # Lire le fichier en tant qu'image
        image = Image.open(BytesIO(file.read())).convert('RGB')
        image = image.resize((32, 32))  # Redimensionner si nécessaire
        image_array = np.array(image) / 255.0  # Normalisation
        image_array = np.expand_dims(image_array, axis=0)

        # Prédire la classe
        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': prediction.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
