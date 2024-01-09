import os
import re
from transformers import pipeline

class Question_Answer_Gen:
    def __init__(self, file_name):
        self.content = self.read_content(file_name)

    def read_content(self, file_name):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_name)
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    def question_generation(self):
        # Split the content into words
        words = re.findall(r'\b\w+\b', self.content)

        chunk_size = 100
        text_sampler = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            text_sampler.append(chunk)

        model_name = "valhalla/t5-base-e2e-qg"

        nlp = pipeline("text2text-generation", model=model_name, tokenizer=model_name)

        generated_questions = []
        for text_sample in text_sampler:
            prompt = "generate question: " + text_sample
            output = nlp(prompt)
            output_question = output[0].get('generated_text')
            question = output_question.split("<sep>")
            generated_questions.append(question[0])
        return generated_questions

    def content_aware_answering(self, generated_questions):
        # Define the model repo
        model_name = "abhitopia/question-answer-generation"
        nlp = pipeline("text2text-generation", model=model_name, tokenizer=model_name)

        for question in generated_questions:
            prompt = f"question: {question} context: {self.content} </s>"
            output = nlp(prompt)
            print(f"Question: {question}\nAnswer: {output[0]['generated_text']}\n")

def main():
    file_name = "example.txt"
    QA = Question_Answer_Gen(file_name)
    generated_questions = QA.question_generation()
    QA.content_aware_answering(generated_questions)

if __name__ == "__main__":
    main()
