from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import sys

def load_model(model_name):
    """Load the QA model and tokenizer from Hugging Face."""
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def answer_question(question, document, model_name):
    """Answer a question based on a given document and model name."""
    # Load model and tokenizer
    model, tokenizer = load_model(model_name)
    
    # Tokenize input
    inputs = tokenizer.encode_plus(question, document, return_tensors='pt')
    input_ids = inputs['input_ids'].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the most likely beginning and end of the answer span
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Convert tokens to the answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <document> <question>")
        return
    
    model_name = sys.argv[1]
    document = sys.argv[2]
    question = sys.argv[3]

    # Get the answer
    answer = answer_question(question, document, model_name)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
