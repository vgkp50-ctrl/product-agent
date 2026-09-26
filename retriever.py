import faiss
import pickle

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

index = faiss.read_index(
    "catalog.index"
)

with open("catalog.pkl","rb") as f:
    docs = pickle.load(f)

def retrieve(query, k=3):

    embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        embedding,
        k
    )

    results = []

    for idx in indices[0]:
        results.append(docs[idx])

    return "\n".join(results)