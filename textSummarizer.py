from transformers import LongformerTokenizer, EncoderDecoderModel
import os
# Load model and tokenizer
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096") 

# Specify the article
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'example.txt')
with open(file_path, 'r') as file:
    content = file.read()

# Tokenize and summarize
input_ids = tokenizer(content, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)

# Get the summary from the output tokens
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print summary
print(summary)