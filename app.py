import os
import json
import re
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from authlib.integrations.flask_client import OAuth
from model_inference import predict_condition

# Load environment variables
if os.path.exists('.env'):
    load_dotenv()
else:
    print("Warning: .env file not found. Ensure environment variables are set in the hosting provider.")
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skin_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# OAuth configuration
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID', 'placeholder-id'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', 'placeholder-secret'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
HISTORY_FILE = 'scans_history.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create database tables
with app.app_context():
    db.create_all()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_scan_to_history(scan_data):
    """Saves a new scan record to the JSON history file."""
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []
    
    # Add timestamp
    scan_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, scan_data) # Newest first
    
    # Keep only last 50 scans for performance
    history = history[:50]
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

@app.route('/', methods=['GET'])
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Email format validation
        email_regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w+$'
        if not re.search(email_regex, email):
            flash('Please enter a valid email address.')
            return redirect(url_for('signup'))

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email already registered. Please login.')
            return redirect(url_for('login'))
        
        new_user = User(email=email, name=name)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password.')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/login/google')
def login_google():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def google_callback():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')
    if user_info:
        email = user_info['email']
        name = user_info.get('name', email.split('@')[0])
        user = User.query.filter_by(email=email).first()
        if not user:
            # Create a new user for first-time Google sign-ins
            user = User(email=email, name=name)
            # Google users don't need a local password hashed, 
            # we can set a random high-entropy placeholder
            user.set_password(os.urandom(24).hex())
            db.session.add(user)
            db.session.commit()
        elif not user.name:
            # Update name if it's missing for an existing user
            user.name = name
            db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
    flash('Google authentication failed.')
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    return render_template('index.html')

@app.route('/history', methods=['GET'])
@login_required
def history():
    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history_data = json.load(f)
    return render_template('history.html', scans=history_data)

@app.route('/faq', methods=['GET'])
def faq():
    return render_template('faq.html')

@app.route('/support', methods=['GET', 'POST'])
def support():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # In a real app, you would use Flask-Mail here
        print(f"NEW CONTACT FORM SUBMISSION:")
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Subject: {subject}")
        print(f"Message: {message}")
        
        flash('Your message has been received! Our team will get back to you soon.')
        return redirect(url_for('support'))
    return render_template('support.html')

@app.route('/technology', methods=['GET'])
def technology():
    return render_template('technology.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to filename to avoid collisions
        unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the uploaded file
        file.save(filepath)
        
        try:
            # Perform prediction
            result = predict_condition(filepath)
            
            # Form image path for HTML rendering
            image_url = url_for('static', filename='uploads/' + unique_filename)
            
            # Save to history
            history_record = {
                'disease': result['disease'],
                'confidence': result['confidence'],
                'risk_level': result['risk_level'],
                'image_url': image_url
            }
            save_scan_to_history(history_record)
            
            return render_template('result.html', 
                                   image_url=image_url,
                                   disease=result['disease'],
                                   confidence=result['confidence'],
                                   risk_level=result['risk_level'],
                                   advice=result['advice'],
                                   cautions=result['cautions'],
                                   complications=result['complications'],
                                   solutions=result['solutions'],
                                   doctor_advice=result['doctor_advice'],
                                   symptoms=result['symptoms'],
                                   prevention=result['prevention'],
                                   locations=result['locations'],
                                   diagnosis=result['diagnosis'],
                                   immediate_actions=result['immediate_actions'],
                                   lifestyle=result['lifestyle'],
                                   visual_features=result['visual_features'],
                                   top_3=result['top_3'])
        except Exception as e:
            flash(f"Error occurring during prediction: {e}")
            return redirect(url_for('dashboard'))
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

if __name__ == '__main__':
    # Generating dummy model if it doesn't exist
    if not os.path.exists('models/skin_disease_model.h5'):
        print("Model not found. Running generate_dummy_model.py...")
        import generate_dummy_model
        generate_dummy_model.create_model().save('models/skin_disease_model.h5')
        
    app.run(debug=True, port=5000)
