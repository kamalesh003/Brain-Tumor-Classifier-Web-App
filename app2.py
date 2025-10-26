from flask import Flask, request, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from PIL import Image
import numpy as np
import pickle
import os
import sys

# Ensure UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

# Ensure the database path exists and is absolute
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'secret_key'

db = SQLAlchemy(app)

# ---------------------- MODELS ----------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# Create the database safely
with app.app_context():
    if not os.path.exists(DB_PATH):
        db.create_all()
        print(f"✅ Database created at: {DB_PATH}")
    else:
        print(f"✅ Database already exists: {DB_PATH}")

# ---------------------- MODEL LOADING ----------------------
try:
    with open('hybrid_model.pkl', 'rb') as f:
        hybrid_model = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    hybrid_model = None

# Define constants
input_shape = (150, 150, 3)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image):
    image = image.resize(input_shape[:2])
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ---------------------- ROUTES ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('register.html', error='Email already registered')

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email
            session.permanent = True
            return redirect('/tumor_index')
        else:
            return render_template('login.html', error='Invalid user')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')

@app.route('/tumor_index')
def tumor_index():
    if 'email' not in session:
        return redirect('/login')
    return render_template('tumor_index.html')

@app.route('/pituitary')
def pituitary():
    if 'email' not in session:
        return redirect('/login')
    return render_template('pituitary.html')

@app.route('/glioma')
def glioma():
    if 'email' not in session:
        return redirect('/login')
    return render_template('glioma.html')

@app.route('/meningioma')
def meningioma():
    if 'email' not in session:
        return redirect('/login')
    return render_template('meningioma.html')

@app.route('/contact')
def contact():
    if 'email' not in session:
        return redirect('/login')
    return render_template('contact.html')

@app.route('/model')
def model_ui():
    if 'email' not in session:
        return redirect('/login')
    return render_template('model.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'email' not in session:
        return redirect('/login')
    if 'file' not in request.files:
        return render_template('results.html', error='No file part'), 400

    file = request.files['file']
    if file.filename == '':
        return render_template('results.html', error='No selected file'), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file.stream).convert('RGB')
        image_array = preprocess_image(image)
        image_array_dup = np.copy(image_array)

        if hybrid_model is None:
            return render_template('results.html', error='Model not loaded properly.')

        prediction = hybrid_model.predict([image_array, image_array_dup])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]

        return render_template('results.html', predicted_label=predicted_label)

    return render_template('results.html', error='Invalid file type'), 400

if __name__ == '__main__':
    app.run(debug=True)
