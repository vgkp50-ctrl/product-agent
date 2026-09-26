from transformers import pipeline

from retriever import retrieve

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device_map="auto"
)

while True:

    question = input("\nAsk: ")

    context = retrieve(question)

    prompt = f"""
You are a Product Catalog Assistant.

Catalog Information:
{context}

Question:
{question}

Answer based only on the catalog.
"""

    response = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )

    print(
        response[0]["generated_text"]
    )