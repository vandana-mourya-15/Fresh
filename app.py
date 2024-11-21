import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from keras.utils import load_img, img_to_array
from flask_cors import CORS
import logging

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=r'uploads/best_model.tflite')
    interpreter.allocate_tensors()
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names based on your model
class_names = [ 
    "Fresh Apple", "Fresh Banana", "Fresh Cucumber", "Fresh Okra", "Fresh Oranges", 
    "Fresh Potato", "Fresh Tomato", "Rotten Apple", "Rotten Banana", "Rotten Cucumber",  
    "Rotten Okra", "Rotten Oranges", "Rotten Potato", "Rotten Tomato"
]

def preprocess_image(img_path):
    """Function to preprocess the image for TensorFlow Lite model."""
    try:
        img = load_img(img_path, target_size=(256, 256))  # Adjusted dimensions
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and add batch dimension
        return img_array
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise

def predict_image(img_path):
    """Function to make a prediction using the TensorFlow Lite model."""
    try:
        input_data = preprocess_image(img_path)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data)
        confidence = output_data[0][predicted_class_index]
        predicted_class = class_names[predicted_class_index]
        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise ValueError(f"Error during prediction: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Predict endpoint called.")
    try:
        if 'file' not in request.files:
            logging.error("No file in request.")
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            logging.error("No file selected for uploading.")
            return jsonify({"error": "No file selected for uploading"}), 400

        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            logging.error("Unsupported file type.")
            return jsonify({"error": "Unsupported file type"}), 400

        # Save file temporarily
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Run prediction
        result = predict_image(file_path)

        # Clean up
        os.remove(file_path)

        return jsonify({
            "predicted_class": result['class'],
            "confidence": f"{result['confidence']:.2f}"
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Internal server error: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Get the port from environment variable or use default (5000)
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"Starting app on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)


