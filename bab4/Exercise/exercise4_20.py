# Exercise 4.20
# Modified Example 4.24: Choose a different piece of text and ask different questions.

from transformers import pipeline

question_answerer = pipeline('question-answering')

context_text = """
AlexNet was designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton. 
It won the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012, achieving a top-five error rate of 15.3%.
The network had eight layers, including five convolutional layers and three fully connected layers.
"""

questions = [
    "Who designed AlexNet?",
    "When did AlexNet win the ImageNet challenge?",
    "What top-five error rate did AlexNet achieve?",
    "How many layers does AlexNet have?"
]

print("=== Question Answering Results ===")
for q in questions:
    ans = question_answerer({'question': q, 'context': context_text})
    print(f"Q: {q}\n -> Answer: '{ans['answer']}' (Score: {ans['score']:.4f})\n")
