import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


# -----------------------------
# 1. Load Gemini API key
# -----------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# -----------------------------
# 2. Load saved FAISS + chunks
# -----------------------------

index = faiss.read_index("vectorstore/index.faiss")

with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# -----------------------------
# 3. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 4. Ask question
# -----------------------------

question = input("\nAsk a question: ")


# -----------------------------
# 5. Convert question to vector
# -----------------------------

question_embedding = model.encode([question])

question_embedding = np.array(question_embedding).astype("float32")


# -----------------------------
# 6. Search FAISS
# -----------------------------

k = 4

distances, indices = index.search(
    question_embedding,
    k
)


# -----------------------------
# 7. Get relevant chunks
# -----------------------------

relevant_chunks = []

for i in indices[0]:
    relevant_chunks.append(chunks[i])


# -----------------------------
# 8. Create context
# -----------------------------

context = "\n\n".join(relevant_chunks)


# -----------------------------
# 9. Send context to Gemini
# -----------------------------

prompt = f"""
You are a college rules assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context, say:

"I don't know based on the provided document."

Do not invent or assume information.

Context:
{context}

Question:
{question}
"""


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)


# -----------------------------
# 10. Display answer
# -----------------------------

print("\nAnswer:")
print(response.text)