import cv2
from ultralytics import YOLO
import time

class WebcamDetector:
    def __init__(self, ruta_modelo):
        self.model = YOLO(ruta_modelo)
        self.cap = cv2.VideoCapture(0)
        self.fps = 0
        self.prev_time = 0
        
    def run(self):
        print("Iniciando detección... Presiona 'q' para salir")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error al capturar frame")
                break
            
            # Calculamos FPS
            current_time = time.time()
            self.fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            
            # Realizamos detección
            resultados = self.model(frame, conf=0.5, verbose=False)
            
            # Dibujar detecciones
            annotated_frame = resultados[0].plot()
            
            # Mostrar FPS
            cv2.putText(annotated_frame, f'FPS: {int(self.fps)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar información de detecciones
            detections = len(resultados[0].boxes) if resultados[0].boxes else 0
            cv2.putText(annotated_frame, f'Detecciones: {detections}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('Proyecto Vision IA Barbijos', annotated_frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Usar el detector
if __name__ == "__main__":
    # Ajusta la ruta según donde se guardó tu modelo
    # Generalmente está en: runs/detect/train/weights/best.pt
    detector = WebcamDetector('runs/detect/train5/weights/best.pt')
    
    try:
        detector.run()
    finally:
        detector.release()