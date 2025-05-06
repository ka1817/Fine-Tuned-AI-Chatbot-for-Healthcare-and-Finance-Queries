import os
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import uvicorn

# --- AWS S3 & Model Setup ---
bucket_name = 'fine-tuned-model1'
prefix = 't5-finetuned/'
local_model_dir = './t5-finetuned'
os.makedirs(local_model_dir, exist_ok=True)

s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

for obj in response.get('Contents', []):
    key = obj['Key']
    file_name = key[len(prefix):]
    if file_name:
        local_path = os.path.join(local_model_dir, file_name)
        if not os.path.exists(local_path):  # avoid re-downloading
            s3.download_file(bucket_name, key, local_path)

tokenizer = T5Tokenizer.from_pretrained(local_model_dir)
model = T5ForConditionalGeneration.from_pretrained(local_model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- FastAPI App ---
app = FastAPI(title="T5 Fine-Tuned Model API")

class InputText(BaseModel):
    text: str
    max_length: int = 100

@app.post("/predict/")
async def predict(input_data: InputText):
    input_text = input_data.text.strip().lower()
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=input_data.max_length,
            num_beams=4,
            early_stopping=True
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"response": output}

if __name__=="__main__":
    uvicorn.run(app,host='0.0.0.0',port=2100)