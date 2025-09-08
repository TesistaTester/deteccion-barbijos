import pandas as pd
import matplotlib.pyplot as plt

# Cargar métricas
df = pd.read_csv("runs/detect/train4/results.csv")

# Graficar pérdida
plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolución de la pérdida")
plt.show()
