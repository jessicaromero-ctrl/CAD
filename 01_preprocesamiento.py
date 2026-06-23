"""
PLN - Ecommerce Text Classification
Etapa 1: Preprocesamiento y extracción de características (en paralelo)
  - TfidfVectorizer  -> keywords / representación vectorial del texto
  - TextBlob         -> polaridad y subjetividad (sentimiento)
  - spaCy            -> entidades nombradas (reglas de dominio, ver nota)

NOTA METODOLÓGICA (transparencia):
El entorno de ejecución no tiene acceso de red a los CDN de modelos de
spaCy (en_core_web_sm), por lo que el NER estadístico no puede descargarse.
En su lugar se usa el motor de spaCy (tokenizer, Matcher, EntityRuler) con
un pipeline basado en reglas, diseñado específicamente para descripciones
de producto: TALLA/MEDIDA, CANTIDAD, MATERIAL, COLOR y MARCA-like tokens.
Esto es, de hecho, más informativo que NER genérico (PERSON/GPE/ORG) para
este dominio, ya que las descripciones de e-commerce rara vez contienen
personas o lugares, pero sí abundan medidas, materiales y colores.
"""
import re
import time
import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import nltk
from nltk.corpus import stopwords
import pickle

t0 = time.time()
STOP = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
df = pd.read_csv("ecommerce_clean.csv")
print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df['category'].nunique()} clases")

# ---------------------------------------------------------------------------
# 2. Limpieza textual básica
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9.\s]", " ", text)   # conserva dígitos y puntos (medidas: 8.7 x 8.7 inch)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["description"].apply(clean_text)

def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in STOP and len(w) > 1)

df["text_no_stop"] = df["clean_text"].apply(remove_stopwords)
print(f"[INFO] Limpieza textual completada ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# 3a. Extracción de características - TfidfVectorizer (keywords)
# ---------------------------------------------------------------------------
t1 = time.time()
tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(df["text_no_stop"])
print(f"[INFO] TF-IDF: matriz {X_tfidf.shape} ({time.time()-t1:.1f}s)")

# top keywords globales (para el reporte)
sums = np.asarray(X_tfidf.sum(axis=0)).ravel()
top_idx = sums.argsort()[::-1][:20]
feat_names = np.array(tfidf.get_feature_names_out())
top_keywords = list(zip(feat_names[top_idx], sums[top_idx].round(2)))

# ---------------------------------------------------------------------------
# 3b. Extracción de características - TextBlob (sentimiento)
# ---------------------------------------------------------------------------
t2 = time.time()
def sentiment_scores(text):
    tb = TextBlob(text)
    return tb.sentiment.polarity, tb.sentiment.subjectivity

sent = df["clean_text"].apply(sentiment_scores)
df["sent_polarity"] = sent.apply(lambda x: x[0])
df["sent_subjectivity"] = sent.apply(lambda x: x[1])
print(f"[INFO] TextBlob sentimiento calculado ({time.time()-t2:.1f}s)")
print(f"        polaridad media={df['sent_polarity'].mean():.4f}  "
      f"subjetividad media={df['sent_subjectivity'].mean():.4f}")

# ---------------------------------------------------------------------------
# 3c. Extracción de características - spaCy (entidades basadas en reglas)
# ---------------------------------------------------------------------------
t3 = time.time()
nlp = spacy.blank("en")
matcher = Matcher(nlp.vocab)

MATERIALS = ["cotton", "leather", "plastic", "metal", "wood", "wooden", "steel",
             "glass", "ceramic", "silk", "synthetic", "rubber", "paper", "cardboard",
             "polyester", "nylon", "aluminium", "aluminum", "brass", "fabric"]
COLORS = ["black", "white", "red", "blue", "green", "yellow", "pink", "grey", "gray",
          "brown", "purple", "orange", "silver", "gold", "beige", "multicolor"]

matcher.add("MEASURE", [[{"LIKE_NUM": True}, {"IS_PUNCT": True, "OP": "?"},
                          {"LIKE_NUM": True, "OP": "?"},
                          {"LOWER": {"IN": ["x", "inch", "inches", "cm", "mm", "ft",
                                            "kg", "gm", "g", "ml", "l", "oz", "lb"]}}]])
matcher.add("QUANTITY", [[{"LOWER": "set"}, {"LOWER": "of"}, {"LIKE_NUM": True}],
                          [{"LIKE_NUM": True}, {"LOWER": {"IN": ["pcs", "pieces", "pack", "packs"]}}]])
matcher.add("MATERIAL", [[{"LOWER": {"IN": MATERIALS}}]])
matcher.add("COLOR", [[{"LOWER": {"IN": COLORS}}]])

def extract_entities(text):
    doc = nlp(text)
    matches = matcher(doc)
    counts = {"MEASURE": 0, "QUANTITY": 0, "MATERIAL": 0, "COLOR": 0}
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        counts[label] += 1
    return counts["MEASURE"], counts["QUANTITY"], counts["MATERIAL"], counts["COLOR"]

ent_results = df["clean_text"].apply(extract_entities)
df["ent_measure"] = ent_results.apply(lambda x: x[0])
df["ent_quantity"] = ent_results.apply(lambda x: x[1])
df["ent_material"] = ent_results.apply(lambda x: x[2])
df["ent_color"] = ent_results.apply(lambda x: x[3])
df["ent_total"] = df[["ent_measure", "ent_quantity", "ent_material", "ent_color"]].sum(axis=1)
print(f"[INFO] spaCy entidades extraídas ({time.time()-t3:.1f}s)")
print(f"        entidades/doc promedio: {df['ent_total'].mean():.3f}")

# ---------------------------------------------------------------------------
# 4. Persistencia de artefactos para la etapa de clasificación
# ---------------------------------------------------------------------------
extra_cols = ["sent_polarity", "sent_subjectivity",
              "ent_measure", "ent_quantity", "ent_material", "ent_color", "ent_total"]
df[["category"] + extra_cols].to_csv("features_extra.csv", index=False)
sparse.save_npz("features_tfidf.npz", X_tfidf)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
df.to_csv("ecommerce_features.csv", index=False)

print(f"\n[OK] Top 20 keywords TF-IDF (peso acumulado):")
for kw, w in top_keywords:
    print(f"     {kw:25s} {w:8.2f}")

print(f"\n[DONE] Etapa de preprocesamiento/extracción finalizada en {time.time()-t0:.1f}s")
