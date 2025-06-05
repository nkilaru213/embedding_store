from embedding_store import create_index
import pickle

# Dummy text data (replace with actual extracted docs)
texts = [
    "Budget for Q2 was finalized in the last meeting.",
    "Project Apollo is scheduled to launch in August.",
    "Customer satisfaction scores improved by 15% this year."
]

model, index, text_map = create_index(texts)

# Save FAISS index
faiss.write_index(index, "index.faiss")

# Save text_map
with open("text_map.pkl", "wb") as f:
    pickle.dump(text_map, f)

print("Embeddings and index stored successfully.")