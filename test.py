# test_toxicity.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 1. Definir etiquetas del reto Kaggle
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# 2. Cargar modelo entrenado
model_path = "./toxicity-multilabel"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Usamos pipeline para predicción
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1  # Usa GPU si hay
)

print("Modelo cargado para predicción")

# 3. Cargar test.csv
test_df = pd.read_csv("./test.csv")

# Si Kaggle test.csv no tiene etiquetas, solo tiene ["id","comment_text"]
if "comment_text" not in test_df.columns:
    raise ValueError("El test.csv debe contener la columna 'comment_text'.")

# 4. Generar predicciones
preds = []
print("Generando predicciones...")
for text in test_df["comment_text"].tolist():
    result = classifier(text)[0]  # lista de dicts [{label, score}, ...]
    # Ordenamos por índice para mapear bien
    scores = [round(result[i]["score"], 4) for i in range(len(label_cols))]
    preds.append(scores)

# Convertir a DataFrame
preds_df = pd.DataFrame(preds, columns=label_cols)

# 5. Crear archivo submission.csv
submission = pd.concat([test_df["id"], preds_df], axis=1)
submission.to_csv("submission.csv", index=False)

print("\n¡Predicciones guardadas en submission.csv!")
print(submission.head())
