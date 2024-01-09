# Input for a text file named example.txt in the same directory
#For Omnistudy, we'll access the document differently and also break it down to something around 50-100 word chunks for each question
import os
import re
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'example.txt')
with open(file_path, 'r') as file:
    content = file.read()

# Split the content into words
words = re.findall(r'\b\w+\b', content)

chunk_size = 100
text_sampler = []
for i in range(0, len(words), chunk_size):
    chunk = ' '.join(words[i:i + chunk_size])
    text_sampler.append(chunk)


# Import generic wrappers
from transformers import pipeline

# Define the model repo
model_name = "valhalla/t5-base-e2e-qg" 

# Create a text-to-text pipeline
nlp = pipeline("text2text-generation", model=model_name, tokenizer=model_name)

# Provide a prompt
generated_questions=[]
for text_sample in text_sampler:
    prompt = "generate question: " + text_sample
    output = nlp(prompt)
    output_question = output[0].get('generated_text')
    question = output_question.split("<sep>")
    generated_questions.append(question[0])


print(generated_questions)
import QA
