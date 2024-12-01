from flask import Flask, render_template, request, jsonify
import os
import cv2
import pickle
import numpy as np

app = Flask(__name__)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

with open("human_detection_model.pkl", "rb") as file:
    clf = pickle.load(file)

def detect_humans(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(gray_image, winStride=(8, 8), padding=(8, 8), scale=1.05)
    human_count = len(boxes)
    return human_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})


    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)
    print(f"Image saved to {image_path}")  

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'success': False, 'message': 'Failed to load image'})

    human_count = detect_humans(image)
    print(f"Detected {human_count} humans in the image.")  

    prediction = clf.predict([[human_count]])  
    human_present = bool(prediction[0])  

    result = {
        "success": True,
        "human_count": human_count,
        "human_present": human_present,
        "message": "Humans detected in the image." if human_present else "No humans detected in the image."
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
