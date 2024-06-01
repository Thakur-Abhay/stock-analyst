from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = "t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
result = generator("You are a smart financial agent. Your task is to generate a readable text after some analysis on the content that has been providede to you from the enriched context that has been provided to you: enriched_context: Birla Soft share value was 250million in 2020, 259 million in 2021 and 400 million in 2023", max_length=50)
print(result[0]['generated_text'])
