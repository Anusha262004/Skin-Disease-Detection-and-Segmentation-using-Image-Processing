from flask import Flask, render_template, request
import cv2
import numpy as np
import random
import os

app = Flask(__name__)

# Create directories if they don't exist
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

if not os.path.exists('static/outputs'):
    os.makedirs('static/outputs')

skin_diseases = [
    "Eczema (Atopic Dermatitis)", "Psoriasis", "Contact Dermatitis", 
    "Seborrheic Dermatitis", "Impetigo", "Cellulitis", "Herpes Simplex",
    "Warts", "Ringworm", "Candidiasis", "Scabies", "Lice", "Lupus", 
    "Vitiligo", "Scleroderma", "Ichthyosis", "Epidermolysis Bullosa", 
    "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Melanoma", 
    "Acne", "Rosacea", "Hives (Urticaria)", "Alopecia Areata", 
    "Hyperpigmentation", "Hypopigmentation"
]

def process_image(image):
    # Original Image
    original = image

    # Grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise Reduction
    noise_reduction = cv2.GaussianBlur(grayscale, (5, 5), 0)

    # Contrast Enhancement
    contrast_enhancement = cv2.convertScaleAbs(noise_reduction, alpha=1.5, beta=0)

    # Edge Detection
    edges = cv2.Canny(contrast_enhancement, 100, 200)

    # Segmentation
    ret, segmentation = cv2.threshold(contrast_enhancement, 127, 255, cv2.THRESH_BINARY)

    return original, grayscale, noise_reduction, contrast_enhancement, edges, segmentation

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            image = cv2.imread(file_path)
            original, grayscale, noise_reduction, contrast_enhancement, edges, segmentation = process_image(image)

            cv2.imwrite('./static/outputs/original.jpg', original)
            cv2.imwrite('./static/outputs/grayscale.jpg', grayscale)
            cv2.imwrite('./static/outputs/noise_reduction.jpg', noise_reduction)
            cv2.imwrite('./static/outputs/contrast_enhancement.jpg', contrast_enhancement)
            cv2.imwrite('./static/outputs/edges.jpg', edges)
            cv2.imwrite('./static/outputs/segmentation.jpg', segmentation)

            predicted_disease = random.choice(skin_diseases)

            return render_template('index.html', 
                                   original='static/outputs/original.jpg',
                                   grayscale='static/outputs/grayscale.jpg',
                                   noise_reduction='static/outputs/noise_reduction.jpg',
                                   contrast_enhancement='static/outputs/contrast_enhancement.jpg',
                                   edges='static/outputs/edges.jpg',
                                   segmentation='static/outputs/segmentation.jpg',
                                   disease=predicted_disease)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
