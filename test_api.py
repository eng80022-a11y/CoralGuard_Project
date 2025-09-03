
import requests

# لينك السيرفر  اللي شغال
url = "http://127.0.0.1:5000/predict"

# الرابط المباشر للصورة
data = {"image_url": "https://res.cloudinary.com/dcvcwzcod/image/upload/v1756902862/CoralGuard/fmwiz6zplrjzk8kbkrbz.jpg"}

# إرسال POST request للـ API
response = requests.post(url, json=data)

# طباعة النتيجة
print(response.json())