from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

# Load CSV
df = pd.read_csv("app/models/disease.csv")  # Update path to your disease.csv

# Vectorize symptoms
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(df['Symptoms']).toarray()

# Create FAISS index
index = faiss.IndexFlatL2(symptom_vectors.shape[1])
index.add(np.array(symptom_vectors))

chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("message")

    # Add to history
    chat_history.append({"user": user_query})

    # FAISS retrieval
    query_vec = vectorizer.transform([user_query]).toarray()
    D, I = index.search(query_vec, k=3)
    retrieved_info = "\n".join(
        f"Disease: {df.iloc[i]['Disease']}\nTreatment: {df.iloc[i]['Treatment']}\nRecommendation: {df.iloc[i]['Recommendation']}\n"
        for i in I[0]
    )

    # Generate response using the retrieved information
    response = (
        f"Based on your symptoms and our medical database, here's what I found:\n\n"
        f"{retrieved_info}\n\n"
        f"Please note that this is a preliminary assessment. For accurate diagnosis and treatment, "
        f"please consult with a healthcare professional."
    )

    chat_history.append({"bot": response})
    return jsonify({"response": response, "history": chat_history})

if __name__ == "__main__":
    app.run(port=5003) 