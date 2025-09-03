
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO

# ----- 1 إنشاء التطبيق -----
app = Flask(__name__)

# ----- 2 تحميل المودل مرة واحدة عند تشغيل السيرفر -----
model_path = "model/Coral_mobilenet_model_best_final.keras"
model = load_model(model_path)

# ----- 3 الكلاسات -----
classes = ["Healthy", "Bleached", "Dead"]

# ----- 4 تعريف الـ endpoint -----
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "image_url" not in data:
        return jsonify({"error": "No image_url provided"}), 400

    img_url = data["image_url"]

    # تحميل الصورة من اللينك
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((128, 128))  # حجم المودل
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    except:
        return jsonify({"error": "Unable to load image from URL"}), 400

    # التنبؤ
    pred = model.predict(img_array)
    pred_index = np.argmax(pred)
    pred_class = classes[pred_index]
    confidence = float(pred[0][pred_index])

    # تحضير JSON
    result_json = {
        "image_name": img_url,
        "prediction": pred_class,
        "confidence": confidence
    }

    return jsonify(result_json)

# ----- 5 تشغيل السيرفر -----
if __name__ == "__main__":
    app.run(debug=True, port=5000)
