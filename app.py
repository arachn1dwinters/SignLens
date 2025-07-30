from flask import Flask, request, render_template
import os
from ImageClassifier import ImageClassifier
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "model/model.pth"
MODEL_URL = "https://drive.google.com/file/d/1EDzbNdQr0q4NDPIF1k_Tywt9Jio5ETid/view?usp=drive_link"

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        if not os.path.exists(file_path):
            image.save(file_path)

        resultDict = {
            "prediction": classifier.classifyImage(file_path),
            "imageSrc": file_path
        }
        return render_template('index.html', result=resultDict)
    else:
        return render_template('index.html')
    
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model file...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")

download_model()

classifier = ImageClassifier()
