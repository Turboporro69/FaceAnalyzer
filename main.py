import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace

font = cv2.FONT_HERSHEY_SIMPLEX

def analyze_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return
    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(color_img, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))
    prediction = DeepFace.analyze(color_img)
    if isinstance(prediction, list):
        prediction = prediction[0]
    for (x, y, u, v) in faces:
        cv2.rectangle(color_img, (x, y), (x + u, y + v), (0, 0, 225), 2)
        emotion_text = f"Emotion: {prediction['dominant_emotion']}"
        race_text = f"Race: {prediction['dominant_race']}"
        cv2.putText(color_img, emotion_text, (x + u + 10, y), font, 1.5, (255, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(color_img, race_text, (x + u + 10, y + 60), font, 1.5, (255, 0, 0), 5, cv2.LINE_AA)
    return color_img

def show_image(img):
    # Obtener el tamaño del panel
    panel_width = panel.winfo_width()
    panel_height = panel.winfo_height()

    # Obtener el tamaño de la imagen
    img_height, img_width, _ = img.shape

    # Calcular la relación de aspecto
    aspect_ratio = img_width / img_height

    # Calcular el nuevo tamaño manteniendo la relación de aspecto
    if panel_width / panel_height > aspect_ratio:
        new_height = panel_height
        new_width = int(panel_height * aspect_ratio)
    else:
        new_width = panel_width
        new_height = int(panel_width / aspect_ratio)

    # Redimensionar la imagen
    img = cv2.resize(img, (new_width, new_height))
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = analyze_image(file_path)
        show_image(img)

def update_webcam():
    ret, frame = cap.read()
    if ret:
        color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(color_img, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))
        if len(faces) > 0:
            prediction = DeepFace.analyze(color_img)
            if isinstance(prediction, list):
                prediction = prediction[0]
            for (x, y, u, v) in faces:
                cv2.rectangle(color_img, (x, y), (x + u, y + v), (0, 0, 225), 2)
                emotion_text = f"Emotion: {prediction['dominant_emotion']}"
                race_text = f"Race: {prediction['dominant_race']}"
                cv2.putText(color_img, emotion_text, (x + u + 10, y), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(color_img, race_text, (x + u + 10, y + 60), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        show_image(color_img)
    panel.after(10, update_webcam)

def start_webcam_mode():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error")
        return
    update_webcam()

def on_resize(event):
    if panel.image:
        show_image(panel.image)

root = tk.Tk()
root.title("Face Analyzer")

button_frame = tk.Frame(root)
button_frame.pack(side="bottom", fill="x", padx=10, pady=10)

btn_upload = tk.Button(button_frame, text="Upload Image", command=upload_image)
btn_upload.pack(side="left", padx=10, pady=10)

btn_webcam = tk.Button(button_frame, text="Webcam Mode", command=start_webcam_mode)
btn_webcam.pack(side="right", padx=10, pady=10)

panel_frame = tk.Frame(root)
panel_frame.pack(fill="both", expand=True)

panel = tk.Label(panel_frame)
panel.pack(fill="both", expand=True)

root.bind('<Configure>', on_resize)

root.mainloop()