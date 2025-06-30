from pymongo import MongoClient
from config import Config

def check_database():
    try:
        # Connect to MongoDB
        client = MongoClient(Config.MONGO_URI)
        db = client.get_database()
        
        # Check if users collection exists and has any documents
        if 'users' in db.list_collection_names():
            users = list(db.users.find())
            print(f"Found {len(users)} users in the database:")
            for user in users:
                print(f"Email: {user['email']}, Role: {user['role']}")
        else:
            print("No users collection found in the database")
            
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")

if __name__ == "__main__":
    check_database() 