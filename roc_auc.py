import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc

# ==============================
# 1. Cargar modelo entrenado
# ==============================
model = YOLO("runs/detect/train4/weights/best.pt")  # Ajusta la ruta a tu modelo

# ==============================
# 2. Definir dataset de test
# ==============================
test_images = "facemask-dataset/test/images"
test_labels = "facemask-dataset/test/labels"

# ==============================
# 3. Inicializar listas
# ==============================
y_test = []
y_scores = []

# ==============================
# 4. Iterar imágenes de test
# ==============================
for img_path in glob.glob(os.path.join(test_images, "*.jpg")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(test_labels, base + ".txt")

    # Saltar si no hay anotación
    if not os.path.exists(label_path):
        continue

    # Leer etiqueta real (asumimos 1 objeto por imagen para simplicidad)
    with open(label_path, "r") as f:
        line = f.readline().strip().split()
        true_class = int(line[0])  # 0 = mask, 1 = no_mask (ajusta según dataset.yaml)

    # Predicción del modelo
    results = model.predict(img_path, conf=0.001, verbose=False)
    boxes = results[0].boxes

    if len(boxes) > 0:
        # Tomar la predicción con mayor confianza
        conf = float(boxes.conf[0])
        pred_class = int(boxes.cls[0])

        # Definir clase positiva como "mask" (id=0)
        score = conf if pred_class == 0 else 1 - conf

        y_test.append(1 if true_class == 0 else 0)  # 1=mask, 0=no_mask
        y_scores.append(score)

# ==============================
# 5. Calcular ROC y AUC
# ==============================
y_test = np.array(y_test)
y_scores = np.array(y_scores)

fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# ==============================
# 6. Graficar curva ROC
# ==============================
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Detección de Barbijos")
plt.legend(loc="lower right")
plt.show()
