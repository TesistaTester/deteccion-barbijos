import cv2
from ultralytics import YOLO
import time

class WebcamDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
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
            
            # Calcular FPS
            current_time = time.time()
            self.fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            
            # Realizar detección
            results = self.model(frame, conf=0.5, verbose=False)
            
            # Dibujar detecciones
            annotated_frame = results[0].plot()
            
            # Mostrar FPS
            cv2.putText(annotated_frame, f'FPS: {int(self.fps)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar información de detecciones
            detections = len(results[0].boxes) if results[0].boxes else 0
            cv2.putText(annotated_frame, f'Detecciones: {detections}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # SEMÁFORO
            semaforo_color = (0, 255, 0) if detections > 0 else (0, 0, 255)  # Verde o Rojo
            semaforo_texto = "CON BARBIJO" if detections > 0 else "SIN BARBIJO"
            
            # cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 5), semaforo_color, -1)
            
            # Dibujar círculo de semáforo en esquina superior derecha
            cv2.circle(annotated_frame, (frame.shape[1] - 30, 30), 15, semaforo_color, -1)
            cv2.circle(annotated_frame, (frame.shape[1] - 30, 30), 15, (255, 255, 255), 2)
            
            # Mostrar texto de estado
            cv2.putText(annotated_frame, f'Estado: {semaforo_texto}', 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, semaforo_color, 2)
            
            # Dibujar recuadro de fondo para mejor visibilidad
            cv2.rectangle(annotated_frame, (5, 85), (300, 130), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (5, 85), (300, 130), semaforo_color, 2)
            
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