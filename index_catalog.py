import pandas as pd
import pickle
import faiss

from sentence_transformers import SentenceTransformer

df = pd.read_csv("catalog.csv")

documents = []

for _, row in df.iterrows():

    text = f"""
    SKU: {row.sku}
    Product: {row.name}
    Category: {row.category}
    Brand: {row.brand}
    Price: ${row.price}
    Description: {row.description}
    """

    documents.append(text)

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

embeddings = model.encode(
    documents,
    convert_to_numpy=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, "catalog.index")

with open("catalog.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Index created")