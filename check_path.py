from app import app, db
import os

with app.app_context():
    uri = app.config['SQLALCHEMY_DATABASE_URI']
    print(f"Config URI: {uri}")
    # In Flask-SQLAlchemy, the database file location can be tricky
    # Let's see what the engine says
    from sqlalchemy import inspect
    engine = db.engine
    print(f"Engine URL: {engine.url}")
    
    # Try to find where it's actually looking
    expected_path = os.path.join(app.instance_path, 'skin_disease.db')
    print(f"Instance path database: {expected_path}")
    print(f"Instance path exists: {os.path.exists(app.instance_path)}")
    if os.path.exists(expected_path):
        print(f"Database found in instance path.")
    else:
        print(f"Database NOT found in instance path.")
        
    root_path = os.path.join(os.getcwd(), 'skin_disease.db')
    print(f"Root path database: {root_path}")
    if os.path.exists(root_path):
        print(f"Database found in root path.")
    else:
        print(f"Database NOT found in root path.")
