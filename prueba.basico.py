import cv2
from ultralytics import YOLO
import numpy as np

# Cargar tu modelo entrenado
model = YOLO('runs/detect/train4/weights/best.pt')  # Ajusta la ruta según donde guardó tu modelo

# Inicializar cámara web
cap = cv2.VideoCapture(0)  # 0 para cámara por defecto

while True:
    # Capturar frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar predicción
    results = model(frame, conf=0.5)  # conf: confianza mínima
    
    # Mostrar resultados
    annotated_frame = results[0].plot()  # Frame con detecciones
    
    # Mostrar frame
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()