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
    scans = db.relationship('Scan', backref='user', lazy=True)
    appointments = db.relationship('Appointment', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Scan Model
class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    image_url = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Chat Message Model
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String(20), nullable=False) # 'User' or 'Doctor'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Appointment Model
class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False) # 'Video', 'Chat', 'Follow-up'
    status = db.Column(db.String(20), default='Scheduled') # 'Scheduled', 'Completed', 'Cancelled'
    date_time = db.Column(db.DateTime, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    notes = db.Column(db.Text, nullable=True)

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

def save_scan_to_history(scan_data, user_id):
    """Saves a new scan record to the database."""
    new_scan = Scan(
        disease=scan_data['disease'],
        confidence=scan_data['confidence'],
        risk_level=scan_data['risk_level'],
        image_url=scan_data['image_url'],
        user_id=user_id
    )
    db.session.add(new_scan)
    db.session.commit()

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
    user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    
    # Prepare data for Chart.js
    labels = [scan.timestamp.strftime("%b %d") for scan in reversed(user_scans)]
    confidence_data = [scan.confidence for scan in reversed(user_scans)]
    
    # Simple risk score mapping: High=3, Medium=2, Low=1
    risk_map = {'High': 3, 'Medium': 2, 'Low': 1}
    risk_data = [risk_map.get(scan.risk_level, 1) for scan in reversed(user_scans)]

    return render_template('history.html', scans=user_scans, 
                           labels=json.dumps(labels), 
                           confidence_data=json.dumps(confidence_data),
                           risk_data=json.dumps(risk_data))

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

@app.route('/booking', methods=['GET', 'POST'])
@login_required
def booking():
    if request.method == 'POST':
        apt_type = request.form.get('type')
        date_str = request.form.get('date')
        time_str = request.form.get('time')
        notes = request.form.get('notes')
        
        try:
            # Combine date and time
            dt_str = f"{date_str} {time_str}"
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
            
            new_apt = Appointment(
                type=apt_type,
                date_time=dt,
                user_id=current_user.id,
                notes=notes
            )
            db.session.add(new_apt)
            db.session.commit()
            flash(f'Your {apt_type} consultation has been scheduled successfully!')
            return redirect(url_for('appointments'))
        except Exception as e:
            flash(f'Error booking appointment: {e}')
            return redirect(url_for('booking'))
            
    return render_template('booking.html', today=datetime.now().strftime('%Y-%m-%d'))

@app.route('/appointments', methods=['GET'])
@login_required
def appointments():
    user_apts = Appointment.query.filter_by(user_id=current_user.id).order_by(Appointment.date_time.asc()).all()
    return render_template('appointments.html', appointments=user_apts)

@app.route('/chat', methods=['GET'])
@login_required
def chat():
    # Fetch chat history for the current user
    messages = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.asc()).all()
    
    # If no messages exist, create the initial doctor greeting
    if not messages:
        initial_msg = ChatMessage(
            content="Hello! I'm Dr. Sarah. I've reviewed your recent scans. How can I help you today?",
            sender='Doctor',
            user_id=current_user.id
        )
        db.session.add(initial_msg)
        db.session.commit()
        messages = [initial_msg]
        
    return render_template('chat.html', messages=messages, now=datetime.now().strftime("%I:%M %p"))

@app.route('/chat/send', methods=['POST'])
@login_required
def send_message():
    content = request.form.get('message')
    if not content:
        return json.dumps({'status': 'error', 'message': 'No content'}), 400
        
    # Save user message
    user_msg = ChatMessage(content=content, sender='User', user_id=current_user.id)
    db.session.add(user_msg)
    
    # Simple deterministic "Doctor" response logic
    bot_responses = [
        "I understand. Looking at your data, I recommend monitoring that area for another week.",
        "That's a valid concern. Have you noticed any itching or redness in that specific spot?",
        "I've updated your care plan based on this. Please check the 'Advice' section in your latest scan.",
        "It's good that you're tracking this. Consistency is key for early detection."
    ]
    
    import random
    response_content = random.choice(bot_responses)
    
    doctor_msg = ChatMessage(content=response_content, sender='Doctor', user_id=current_user.id)
    db.session.add(doctor_msg)
    db.session.commit()
    
    return json.dumps({
        'status': 'success',
        'user_message': user_msg.content,
        'doctor_message': doctor_msg.content,
        'timestamp': doctor_msg.timestamp.strftime("%I:%M %p")
    })

@app.route('/quick-chat', methods=['GET'])
@login_required
def quick_chat():
    # Create or find a 'Chat' appointment for today
    # to mock a real scenario where a chat is available
    new_apt = Appointment(
        type='Chat',
        date_time=datetime.now(),
        user_id=current_user.id,
        notes='Quick Direct Chat'
    )
    db.session.add(new_apt)
    db.session.commit()
    return redirect(url_for('chat'))

@app.route('/video-call', methods=['GET'])
@login_required
def video_call():
    return render_template('video_call.html')

@app.route('/predict', methods=['POST'])
@login_required
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
            save_scan_to_history(history_record, current_user.id)
            
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
