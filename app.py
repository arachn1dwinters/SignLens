from flask import Flask, request, render_template
import os
from ImageClassifier import ImageClassifier

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classifier = ImageClassifier()

@app.route("/", methods=['GET', 'POST'])
def hello_world():
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