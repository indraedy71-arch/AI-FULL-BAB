# Exercise 4.18
# Modified Example 4.22: Choose a different piece of text for text analysis.

import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

custom_sentence = """
Deep learning is a subset of machine learning based on artificial neural networks with representation learning. 
Geoffrey Hinton, Yann LeCun, and Yoshua Bengio were awarded the 2018 Turing Award for their breakthroughs in deep learning.
Convolutional Neural Networks excel in computer vision tasks, whereas Transformers have revolutionized Natural Language Processing.
"""

print("=== Analyzing Custom Text ===")
tokens = nltk.word_tokenize(custom_sentence)
print("Tokens:")
print(tokens)

tagged = nltk.pos_tag(tokens)
print("\nPOS Tagged:")
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print("\nNamed Entities:")
print(entities)
