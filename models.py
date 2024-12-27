from extensions import db

class User(db.Model):
    __tablename__ = 'transfer_users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.LargeBinary)
    provider = db.Column(db.String(50))
    provider_user_id = db.Column(db.String(100))