from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import sys

def load_model(model_name):
    """Load the VQA model and processor from Hugging Face."""
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return model, processor

def answer_question(image_path, question, model_name):
    """Answer a question based on a given image and model name."""
    # Load model and processor
    model, processor = load_model(model_name)

    # Open image
    image = Image.open(image_path)

    # Preprocess image and generate inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate output using the model
    question_prompt = f"question: {question}"
    inputs = processor.tokenizer(question_prompt, return_tensors="pt").input_ids

    # Perform inference
    outputs = model.generate(pixel_values=pixel_values, input_ids=inputs)

    # Decode the output
    answer = processor.decode(outputs[0], skip_special_tokens=True)

    return answer

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <image_path> <question>")
        return

    model_name = sys.argv[1]
    image_path = sys.argv[2]
    question = sys.argv[3]

    # Get the answer
    answer = answer_question(image_path, question, model_name)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
