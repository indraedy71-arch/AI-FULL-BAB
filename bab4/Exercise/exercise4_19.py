# Exercise 4.19
# Modified Example 4.23: Evaluate multiple positive and negative sentences using HuggingFace Transformers.

from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = [
    "This movie was absolutely wonderful and inspiring!",
    "The plot was terrible, boring, and a complete waste of time.",
    "The deep learning course is challenging yet highly rewarding.",
    "I hated the poor service and terrible food quality at the restaurant."
]

print("=== Sentiment Analysis Results ===")
for s in sentences:
    res = classifier(s)[0]
    print(f"Sentence: \"{s}\"\n -> Label: {res['label']}, Score: {res['score']:.4f}\n")
