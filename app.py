import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from keras.utils import load_img, img_to_array

app = Flask(__name__)
try:
    interpreter = tf.lite.Interpreter(model_path=r'uploads/best_model.tflite')
    interpreter.allocate_tensors()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the class names based on your model
class_names = [ 
    "Fresh Apple", "Fresh Banana", "Fresh Cucumber", "Fresh Okra", "Fresh Oranges", 
    "Fresh Potato", "Fresh Tomato", "Rotten Apple", "Rotten Banana", "Rotten Cucumber",  
    "Rotten Okra", "Rotten Oranges", "Rotten Potato", "Rotten Tomato"
]

def preprocess_image(img_path):
    """Function to preprocess the image for TensorFlow Lite model."""
    img = load_img(img_path, target_size=(256, 256))  # Adjusted dimensions
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and add batch dimension
    return img_array

def predict_image(img_path):
    """Function to make a prediction using the TensorFlow Lite model."""
    try:
        # Preprocess the image
        input_data = preprocess_image(img_path)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the predicted class
        predicted_class_index = np.argmax(output_data)
        confidence = output_data[0][predicted_class_index]

        # Prepare the result
        predicted_class = class_names[predicted_class_index]
        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Check if the file has a valid image extension
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        # Save the uploaded file to a temporary location
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Make a prediction using TensorFlow Lite model
        result = predict_image(file_path)

        # Remove the file after prediction
        os.remove(file_path)

        return jsonify({
            "predicted_class": result['class'],
            "confidence": f"{result['confidence']:.2f}"
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    # Make sure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Use the PORT environment variable for dynamic port binding (for platforms like Render)
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 for local development
    app.run(debug=True, host='0.0.0.0', port=port)  # Ensure it listens on all interfaces
