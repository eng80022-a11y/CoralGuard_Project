# test_model.py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 1️⃣ تحميل المودل
model = load_model("model/Coral_mobilenet_model_best_final.keras")

# 2️⃣ تحميل صورة للتجربة
img_path = "test_images/coral1.jpg"  # حطي مسار صورتك هنا
img = image.load_img(img_path, target_size=(224,224))  # لو المودل محتاج 224x224
img_array = image.img_to_array(img)/255.0  # تحويل الصورة لمصفوفة وتقسيم القيم على 255
img_array = np.expand_dims(img_array, axis=0)  # إضافة بعد batch dimension

# 3️⃣ التنبؤ
prediction = model.predict(img_array)

# 4️⃣ طباعة النتيجة
print("Prediction array:", prediction)
