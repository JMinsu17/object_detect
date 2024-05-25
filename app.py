from flask import Flask, request, jsonify, send_file
from io import BytesIO
from analysis import analyze_image
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        image_data = file.read()
        result_img = analyze_image(image_data)

        if result_img is None:
            return jsonify({"error": "Image processing failed"})

        # 결과 이미지를 메모리 버퍼에 저장
        _, img_encoded = cv2.imencode('.jpg', result_img)
        img_bytes = BytesIO(img_encoded.tobytes())

        return send_file(img_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
