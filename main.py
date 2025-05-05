import os
import boto3
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# AWS S3 details
bucket_name = 'fine-tuned-model1'
prefix = 't5-finetuned/'
local_model_dir = './t5-finetuned'

os.makedirs(local_model_dir, exist_ok=True)

# Initialize S3 client (uses ~/.aws/credentials or environment variables)
s3 = boto3.client('s3')

# List and download all model files
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

for obj in response.get('Contents', []):
    key = obj['Key']
    file_name = key[len(prefix):]
    if file_name:  # avoid downloading the folder itself
        local_path = os.path.join(local_model_dir, file_name)
        s3.download_file(bucket_name, key, local_path)

# Load model from the local directory
tokenizer = T5Tokenizer.from_pretrained(local_model_dir)
model = T5ForConditionalGeneration.from_pretrained(local_model_dir)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define prediction function
def predict_local(input_text, max_length=100):
    input_text = input_text.strip().lower()
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test
query = "Is it possible to adjust the terms of my loan repayment plan?"
response = predict_local(query)
print("Model response:", response)
