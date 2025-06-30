# MediBuddy - Patient Health Monitoring System

MediBuddy is a comprehensive web application for patient health monitoring that allows patients to track their health records, upload medical reports, and get preliminary disease predictions based on symptoms.

## Features

- User authentication (Patient and Doctor roles)
- Health record management
- Medical report upload and storage
- Disease prediction using ML models
- Responsive and modern UI
- MongoDB database integration

## Prerequisites

- Python 3.8 or higher
- MongoDB
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medibuddy.git
cd medibuddy
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following content:
```
SECRET_KEY=your-secret-key-here
MONGO_URI=mongodb://localhost:27017/medibuddy
```

5. Create necessary directories:
```bash
mkdir -p app/static/uploads
```

## Running the Application

1. Start MongoDB:
```bash
mongod
```

2. Run the Flask application:
```bash
python run.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
MediBuddy/
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── main.py
│   │   └── patient.py
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── uploads/
│   ├── templates/
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── register.html
│   │   └── patient/
│   └── utils/
│       ├── __init__.py
│       └── ml_models.py
├── config.py
├── requirements.txt
└── run.py
```

## Features in Detail

### User Authentication
- Secure login and registration system
- Role-based access control (Patient/Doctor)
- User profile management

### Health Records
- Daily health tracking
- Vital signs monitoring
- Symptoms and medication tracking
- Historical data visualization

### Medical Reports
- PDF and image upload support
- Report categorization
- Secure storage in MongoDB
- Easy access and viewing

### Disease Prediction
- ML-based symptom analysis
- Preliminary disease suggestions
- Treatment recommendations
- Confidence scoring

## Security Features

- Password hashing
- Session management
- File upload validation
- Role-based access control
- Secure MongoDB connection

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask framework
- MongoDB
- scikit-learn
- Bootstrap
- Font Awesome 