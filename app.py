from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
import glob
from PIL import Image
from io import BytesIO
import base64

# Initialize MobileNet model
mobilenet_model = MobileNet(weights='imagenet', include_top=False)

import joblib
import pickle

import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.applications import ResNet50
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import numpy as np
# import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing.sequence import pad_sequences

# Load ResNet50 model + higher level layers
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load the KNN model
knn_model_path = 'models/knn_models.h5'  # Update with the correct file path
knn_models = joblib.load(knn_model_path)

image_caption_model_path = 'models/image_caption_model.h5'  # Update with the correct path
image_caption_model = load_model(image_caption_model_path)

# Optionally, load the pre-extracted features and image paths
X = np.load('npyfiles/features.npy')
image_paths = np.load('npyfiles/image_paths.npy', allow_pickle=True)


# Load tokenizer
with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = base_model.predict(preprocessed_img)
    return features.flatten()

def find_similar_images(img_path, num_results=5):
    query_features = extract_features(img_path)
    _, indices = knn_models.kneighbors([query_features])
    similar_images = ["images/" + os.path.basename(image_paths[idx]) for idx in indices.flatten()]

    # Extracting just the filenames without the path prefix
    similar_image_filenames = [os.path.basename(img) for img in similar_images]

    df = pd.read_csv('Dataset/fashion.csv')
    similar_image_names = df.loc[df['Image'].isin(similar_image_filenames), 'ProductTitle'].tolist()

    return similar_images, similar_image_names





def load_label_encoders():
    label_encoders = {}
    label_encoders['category'] = joblib.load('label_encoders/label_encoder_category.pkl')
    label_encoders['subcategory'] = joblib.load('label_encoders/label_encoder_subcategory.pkl')
    label_encoders['color'] = joblib.load('label_encoders/label_encoder_color.pkl')
    label_encoders['product_type'] = joblib.load('label_encoders/label_encoder_product_type.pkl')
    label_encoders['usage'] = joblib.load('label_encoders/label_encoder_usage.pkl')
    return label_encoders



def load_all_models():
    models = {}
    models['category'] = load_model('models/model_category.h5')
    models['subcategory'] = load_model('models/model_subcategory.h5')
    models['color'] = load_model('models/model_color.h5')
    models['product_type'] = load_model('models/model_product_type.h5')
    models['usage'] = load_model('models/model_usage.h5')
    return models

def get_image_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = mobilenet_model.predict(image)
    return features.reshape(-1)  # Flatten the features

def predict_details(image_path, models):
    features = get_image_features(image_path)
    features = features.reshape(1, -1)
    encoders = load_label_encoders()
    details = {}
    print(encoders['category'])

    details['Category'] = encoders['category'].inverse_transform([np.argmax(models['category'].predict(features))])[0]
    details['SubCategory'] = encoders['subcategory'].inverse_transform([np.argmax(models['subcategory'].predict(features))])[0]
    details['Color'] = encoders['color'].inverse_transform([np.argmax(models['color'].predict(features))])[0]
    details['ProductType'] = encoders['product_type'].inverse_transform([np.argmax(models['product_type'].predict(features))])[0]
    details['Usage'] = encoders['usage'].inverse_transform([np.argmax(models['usage'].predict(features))])[0]
    return details


from sklearn.metrics import pairwise_distances

def preprocess_image1(image_path):
    image = load_img(image_path, target_size=(299, 299)) # Adjust target_size if using ResNet50
    image = img_to_array(image)
    image = preprocess_input(image)
    return image.reshape((1, 299, 299, 3))

def generate_description(model, image_path):
    # Preprocess the image
    image = preprocess_image1(image_path)
    image_features = base_model.predict(image)

    # Start token
    start_token = [1]
    input_sequence = pad_sequences([start_token], maxlen=18)  # Change maxlen to 18

    # Generate the description iteratively
    description = []
    for i in range(18 - 1):  # Keep the loop range as 18 - 1
        predictions = model.predict([image_features, input_sequence], verbose=0)
        predicted_word_index = np.argmax(predictions[0, i, :])
        
        # Translate the predicted word index to a word
        word = tokenizer.index_word.get(predicted_word_index, ' ')
        if word == ' ':
            break
            
        description.append(word)

        # Update the input sequence for the next iteration
        input_sequence = np.append(input_sequence, [[predicted_word_index]], axis=1)
        input_sequence = pad_sequences(input_sequence, maxlen=18)  # Change maxlen to 18

    return ' '.join(description)






app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    # Handle file upload and prediction logic here
    # ...
    # Check if the request has an image file
    if 'file' not in request.files:
        print("File not found in request")  # Additional debug print
        return jsonify(error='No file part'), 400
    file = request.files['file']

    
    # Check if a file was uploaded
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    # Save the uploaded image
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load models and predict details
        models = load_all_models()
        details = predict_details(file_path, models)

        # Optionally, remove the uploaded file after processing
        #
# Generate description for the uploaded image
        description = generate_description(image_caption_model, file_path)  # Replace 'your_model' with the actual model
        print(description)
        
        # Call your function to get the recommended products for the uploaded image
        # Get similar products using the similarity model
        
        #similar_products = similarity_model.get_similar_products(file_path)
        num_results = int(request.form.get('num_results', 5))
        similar_image_paths, similar_image_names = find_similar_images(file_path, num_results)
      
        os.remove(file_path)
        # Extract the image URLs or paths
        #recommended_images = [product['ImageURL'] for idx, product in recommended_products.iterrows()]


        return render_template('index.html', details=details,description = description,similar_images=zip(similar_image_names, similar_image_paths))
    else:
        return jsonify(error='Invalid file type'), 400

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
