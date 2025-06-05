from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index
index = faiss.read_index("index.faiss")

# Load text_map
with open("text_map.pkl", "rb") as f:
    text_map = pickle.load(f)