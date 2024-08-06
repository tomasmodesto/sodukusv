from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Carregar o modelo
model = load_model('best_model.h5')

# Função para pré-processar a imagem
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return imgThreshold

# Função para reordenar pontos para a perspectiva de warp
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# Função para encontrar o maior contorno
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# Função para dividir a imagem em 81 partes
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

# Função para processar cada caixa
def processBox(image):
    img_inverted = cv2.bitwise_not(image)
    return img_inverted

# Função para obter previsões dos dígitos
def getPrediction(boxes, model):
    result = []
    for i, image in enumerate(boxes):
        img_processed = processBox(image)
        img_resized = cv2.resize(img_processed, (28, 28))
        img_normalized = img_resized / 255.0
        img_normalized = img_normalized.reshape(1, 28, 28, 1)
        predictions = model.predict(img_normalized)
        classIndex = np.argmax(predictions)
        probabilityValue = np.max(predictions)
        if probabilityValue > 0.7:
            result.append((classIndex, probabilityValue))
        else:
            result.append((0, 0))
    return result

# Função para processar a imagem recebida
def process_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (450, 450))
    imgThreshold = preProcess(img)
    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, _ = biggestContour(contours)

    if biggest.size != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (450, 450))
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        boxes = splitBoxes(imgWarpColored)
        numbers = getPrediction(boxes, model)
        numbers = np.asarray([num[0] for num in numbers])
        return numbers.tolist()
    else:
        return []

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        numbers = process_image(file)
        return jsonify({"numbers": numbers}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)
