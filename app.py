from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
import numpy as np
from scipy.sparse import hstack
import sqlite3

# LOGIN IMPORTS
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

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
# HOME
# ---------------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    result = None
    score = None

    if request.method == "POST":
        comment = request.form["comment"]

        file = request.files["evidence"]
        if file and file.filename != "":
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))

        result, score = check_comment(comment)

        cursor.execute("INSERT INTO comments VALUES (?, ?, ?)", (comment, result, score))
        conn.commit()

    return render_template("index.html", result=result, score=score)

# ---------------------------
# LOGIN
# ---------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username]["password"] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for("home"))
        else:
            return "Invalid credentials!"

    return render_template("login.html")
# ---------------------------
# HISTORY (PROTECTED)
# ---------------------------
@app.route("/history")
@login_required
def history():
    cursor.execute("SELECT * FROM comments")
    data = cursor.fetchall()
    return render_template("history.html", data=data)

# ---------------------------
# DASHBOARD (PROTECTED)
# ---------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    import plotly.graph_objs as go
    import plotly.io as pio

    cursor.execute("SELECT result, COUNT(*) FROM comments GROUP BY result")
    data_count = cursor.fetchall()

    results = [row[0] for row in data_count]
    counts = [row[1] for row in data_count]

    fig = go.Figure([go.Bar(x=results, y=counts)])
    fig.update_layout(
        title="Bullying vs Safe Comments",
        xaxis_title="Result",
        yaxis_title="Count"
    )

    graph_html = pio.to_html(fig, full_html=False)

    return render_template("dashboard.html", graph_html=graph_html)

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
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)