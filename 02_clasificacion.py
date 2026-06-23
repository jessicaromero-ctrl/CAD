"""
PLN - Ecommerce Text Classification
Etapa 2: Clasificación (en paralelo, considerando todas las columnas derivadas)
  Features combinadas: TF-IDF (texto) + TextBlob (sentimiento) + spaCy (entidades)
  Modelos: RandomForestClassifier, LogisticRegression, SVC
"""
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
import joblib

t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Cargar features de la etapa 1
# ---------------------------------------------------------------------------
df_extra = pd.read_csv("features_extra.csv")
X_tfidf = sparse.load_npz("features_tfidf.npz")

extra_cols = ["sent_polarity", "sent_subjectivity",
              "ent_measure", "ent_quantity", "ent_material", "ent_color", "ent_total"]
X_extra = df_extra[extra_cols].values
scaler = StandardScaler()
X_extra_scaled = scaler.fit_transform(X_extra)

# Matriz final: todas las columnas / features combinadas
X = sparse.hstack([X_tfidf, sparse.csr_matrix(X_extra_scaled)]).tocsr()
le = LabelEncoder()
y = le.fit_transform(df_extra["category"])

print(f"[INFO] Matriz de features final: {X.shape}  (TF-IDF: {X_tfidf.shape[1]} + extra: {X_extra.shape[1]})")
print(f"[INFO] Clases: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[INFO] Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 2. Modelos (entrenamiento/evaluación "en paralelo")
# ---------------------------------------------------------------------------
models = {
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, n_jobs=-1, C=5.0),
    # SVC con kernel lineal: se usa LinearSVC (formulación optimizada de SVC
    # lineal vía liblinear) por escalabilidad sobre ~28k documentos x 4007
    # dimensiones; un SVC(kernel='rbf') estándar es computacionalmente
    # inviable en este volumen (complejidad ~O(n^2)-O(n^3)).
    "SVC (kernel lineal)": LinearSVC(C=1.0, max_iter=5000, dual=True),
}

results = {}
for name, clf in models.items():
    t1 = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    elapsed = time.time() - t1

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)

    results[name] = dict(acc=acc, prec=prec, rec=rec, f1=f1, cm=cm,
                          report=report, time=elapsed)

    print("=" * 78)
    print(f"[MODELO] {name}  (entrenado en {elapsed:.1f}s)")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall    (macro): {rec:.4f}")
    print(f"  F1-score  (macro): {f1:.4f}")
    print("-" * 78)
    print(report)
    print("Matriz de confusión (filas=real, columnas=predicho):")
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    print()

    joblib.dump(clf, f"model_{name.split()[0]}.pkl")

# ---------------------------------------------------------------------------
# 3. Tabla comparativa final
# ---------------------------------------------------------------------------
print("=" * 78)
print("RESUMEN COMPARATIVO DE MODELOS")
print("=" * 78)
summary = pd.DataFrame({
    name: {"Accuracy": r["acc"], "Precision_macro": r["prec"],
           "Recall_macro": r["rec"], "F1_macro": r["f1"], "Tiempo_s": r["time"]}
    for name, r in results.items()
}).T
summary = summary.sort_values("F1_macro", ascending=False)
print(summary.round(4).to_string())

best_model = summary.index[0]
print(f"\n[CONCLUSION] Mejor modelo según F1-macro: {best_model} "
      f"(F1={summary.loc[best_model,'F1_macro']:.4f})")

summary.to_csv("resultados_comparativos.csv")
print(f"\n[DONE] Etapa de clasificación finalizada en {time.time()-t0:.1f}s")
