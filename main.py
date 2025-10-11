# ENTRENAMIENTO MULTI-ETIQUETA CON TOXICITY DATASET (Kaggle)
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

# Deshabilitar wandb y warnings
os.environ["WANDB_DISABLED"] = "true"
import warnings
warnings.filterwarnings('ignore')

print("Iniciando entrenamiento BERT multi-etiqueta...")

# 1. Cargar datos
train_df = pd.read_csv("./train_sample.csv")
test_df = pd.read_csv("./test_sample.csv")

# Etiquetas de Kaggle
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Verificamos que las columnas existan
for col in label_cols:
    if col not in train_df.columns:
        raise ValueError(f"Falta la columna {col} en train.csv")

# Renombramos
train_df = train_df[["id", "comment_text"] + label_cols]
train_df = train_df.rename(columns={"comment_text": "text"})

# Si test.csv no tiene etiquetas, lo dejamos sin ellas
if set(label_cols).issubset(test_df.columns):
    test_df = test_df[["id", "comment_text"] + label_cols].rename(columns={"comment_text": "text"})
else:
    test_df = test_df[["id", "comment_text"]].rename(columns={"comment_text": "text"})

#print(f"Datos cargados: {len(train_df)} train, {len(test_df)} test")

# 2. Configurar modelo
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

print("Modelo BERT configurado para clasificación multi-etiqueta")

# 3. Tokenización
def tokenize_dataset(df, is_train=True):
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    data = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"]
    }
    if is_train:
        data["labels"] = torch.tensor(df[label_cols].values, dtype=torch.float)
    return Dataset.from_dict(data)

train_dataset = tokenize_dataset(train_df, is_train=True)
eval_dataset = tokenize_dataset(test_df, is_train=True) if set(label_cols).issubset(test_df.columns) else None

print("Datos tokenizados")

# 4. Configuración entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=200,
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=2,
    report_to=[],
    remove_unused_columns=False,
)

# 5. Métricas multi-etiqueta
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Aplicamos sigmoid para multi-label
    probs = 1 / (1 + np.exp(-predictions))
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="micro")
    return {"accuracy": acc, "f1_micro": f1}

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if eval_dataset else None,
    compute_metrics=compute_metrics if eval_dataset else None,
)

print("Trainer configurado")

# 7. Entrenar
print("\nINICIANDO ENTRENAMIENTO...")
print("="*50)
trainer.train()
print("\nENTRENAMIENTO COMPLETADO!")

# 8. Evaluar si hay etiquetas en test
if eval_dataset:
    eval_result = trainer.evaluate()
    print(f"\nResultados en test → Accuracy: {eval_result['eval_accuracy']:.4f}, F1: {eval_result['eval_f1_micro']:.4f}")

# 9. Guardar modelo
trainer.save_model("./toxicity-multilabel")
tokenizer.save_pretrained("./toxicity-multilabel")

print("Modelo guardado en ./toxicity-multilabel")

# 10. Prueba rápida
from transformers import pipeline
classifier = pipeline("text-classification", model="./toxicity-multilabel", tokenizer=tokenizer, return_all_scores=True)

test_cases = [
    "You are so stupid and annoying",
    "I love this place, it's amazing!",
    "Kill yourself idiot"
]

print("\nPRUEBA RÁPIDA:")
for text in test_cases:
    result = classifier(text)
    scores = {label_cols[i]: round(r["score"], 3) for i, r in enumerate(result[0])}
    print(f"'{text}' → {scores}")
