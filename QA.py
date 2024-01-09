
def content_aware_Answering(generated_questions, context):
    from transformers import pipeline

    model_name = "abhitopia/question-answer-generation"
    nlp = pipeline("text2text-generation", model=model_name, tokenizer=model_name)

    generated_questions = ['What year did the War of 1812 begin?', 'Who opposed U S colonial settlement in the Old Northwest?', 'When did the U.S. begin to fight?', 'What was the name of the convention held by the Federalists in December?']

    for question in generated_questions:
        prompt = f"question: {question} context: {content} </s>"
        output = nlp(prompt)
        print(f"Question: {question}\nAnswer: {output[0]['generated_text']}\n")