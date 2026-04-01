import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. LOAD DATASET
# ---------------------------
data = pd.read_csv("bullying dataset.csv")

# ---------------------------
# 2. CLEAN DATA
# ---------------------------
data = data.dropna(subset=["text"])

# ---------------------------
# 3. LABEL FIX (IMPORTANT)
# ---------------------------
data["cyberbullying"] = data["cyberbullying"].replace({
    1: "bullying",
    0: "safe"
})

# ---------------------------
# 4. FEATURES
# ---------------------------
X_text = data["text"]
X_extra = data[["toxicity_score", "abusive_words"]].values

y = data["cyberbullying"]

# ---------------------------
# 5. TEXT VECTORIZATION
# ---------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
X_text_vec = vectorizer.fit_transform(X_text)

# ---------------------------
# 6. COMBINE FEATURES
# ---------------------------
X = hstack([X_text_vec, X_extra])

# ---------------------------
# 7. TRAIN-TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# 8. TRAIN MODEL
# ---------------------------
model = LinearSVC()
model.fit(X_train, y_train)

# ---------------------------
# 9. TEST ACCURACY
# ---------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# 10. SAVE MODEL
# ---------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")