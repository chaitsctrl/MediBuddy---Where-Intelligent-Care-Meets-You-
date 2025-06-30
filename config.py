import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '4093583315b77bd2dd43dfe17f4bf613e386b2aeae917ad433665c5606420e38'
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/medibuddy'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/static/uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
    
    # ML Model URLs
    DEMENTIA_MODEL_URL = os.getenv('DEMENTIA_MODEL_URL', 'http://localhost:5003')
    HEART_MODEL_URL = os.getenv('HEART_MODEL_URL', 'http://localhost:5004')
    LUNG_MODEL_URL = os.getenv('LUNG_MODEL_URL', 'http://localhost:5005')
    BRAIN_TUMOUR_MODEL_URL = os.getenv('BRAIN_TUMOUR_MODEL_URL', 'http://localhost:5001') 