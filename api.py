import transformers
import torch
import config

pipeline = transformers.pipeline(
    "text-generation", model=config.model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
model_response = pipeline("Hey how are you doing today?")
