# Example 4.23
# https://github.com/huggingface/transformers
# pip install transformers

from transformers import pipeline

classifier = pipeline('sentiment-analysis')
results = classifier('This is a good movie.')
print(results)
