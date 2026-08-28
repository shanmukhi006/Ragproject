import pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os


# -----------------------------
# 1. Extract text from PDF
# -----------------------------

pdf = pymupdf.open("data/college_rules.pdf")

text = ""

for page in pdf:
    text += page.get_text()


# -----------------------------
# 2. Create chunks
# -----------------------------

chunk_size = 500
overlap = 50

chunks = []

start = 0

while start < len(text):

    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    start = end - overlap


print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Create embeddings
# -----------------------------

print("Creating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings created.")


# -----------------------------
# 4. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Number of vectors:", index.ntotal)


# -----------------------------
# 5. Save FAISS index + chunks
# -----------------------------

os.makedirs("vectorstore", exist_ok=True)
#2. Save the FAISS index
faiss.write_index(
    index,
    "vectorstore/index.faiss"
)
#3. Save the chunks
with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)


print("FAISS index and chunks saved successfully!")