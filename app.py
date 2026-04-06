from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import pickle
import numpy as np
from scipy.sparse import hstack
import sqlite3

from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

# ---------------------------
# DATABASE
# ---------------------------
conn = sqlite3.connect("comments.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS comments(
    text TEXT,
    result TEXT,
    score REAL
)
""")

# ---------------------------
# LOAD MODEL
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------------------
# APP SETUP
# ---------------------------
app = Flask(__name__)
app.secret_key = "secret123"

# ---------------------------
# LOGIN SETUP
# ---------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {
    "admin": {"password": "admin123"}
}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ---------------------------
# ML FUNCTION
# ---------------------------
def check_comment(text):
    text_vec = vectorizer.transform([text])

    abusive_list = [
        "stupid", "idiot", "bitch", "hate", "useless",
        "dumb", "fool", "loser", "trash", "ugly"
    ]

    abusive_words = 1 if any(word in text.lower() for word in abusive_list) else 0
    toxicity_score = 0.8 if abusive_words == 1 else 0.2

    extra = np.array([[toxicity_score, abusive_words]])
    final_input = hstack([text_vec, extra])

    prediction = model.predict(final_input)

    return prediction[0], round(toxicity_score, 2)

# ---------------------------
# HOME (CHAT UI ONLY)
# ---------------------------
@app.route("/")
@login_required
def home():
    return render_template("index.html")

# ---------------------------
# PREDICT (FOR CHAT UI)
# ---------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json()
    text = data.get("text", "")

    result, score = check_comment(text)

    # Save to DB
    cursor.execute("INSERT INTO comments VALUES (?, ?, ?)", (text, result, score))
    conn.commit()

    return jsonify({"result": result, "score": score})

# ---------------------------
# LOGIN
# ---------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username]["password"] == password:
            login_user(User(username))
            return redirect(url_for("home"))
        else:
            return "Invalid credentials!"

    return render_template("login.html")

# ---------------------------
# LOGOUT
# ---------------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)