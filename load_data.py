import re

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def get_datasets():
    dataset = load_dataset("emozilla/sat-reading")

    train_dataset = dataset['train'].to_pandas()
    valid_dataset = dataset['validation'].to_pandas()
    test_dataset = dataset['test'].to_pandas()

    return {
        "train_dataset": train_dataset, 
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset
    }

SYSTEM_PROMPT = """You are a helpful AI assistant developed by META. Respond safely and accurately."""


def extract_sections(text):
    sections = {
        "passenger": "",
        "question": "",
        "choices": [],
        "answer_letter": ""
    }
    
    content = text.split("SAT READING COMPREHENSION TEST")[-1]
    content = content.split("Question")[0].strip()
    
    question = re.split(r'(Question\s+\d+:)', text, flags=re.IGNORECASE)[-1]
    
    question, choices = question.split("A) ")
    
    choices = "A) " + choices
    choices = choices.split("Answer")[0].strip()
    
    choices = choices.split("\n")
    
    sections['passenger'] = content.strip()
    sections['question'] = question.strip()
    sections['choices'] = [c.strip() for c in choices]
    
    return sections
    
def map_answer(text, answer_letter):
    sections = extract_sections(text)
    
    for choice in sections['choices']:
        if choice.startswith(f"{answer_letter})"):
            return choice
    
    return answer_letter

def make_prompt(text, answer_letter):
    sections = extract_sections(text)
    choices_text = "\n".join(sections['choices'])
    
    user_content = f"""
    Read the passenger and answer the question.
    
    ### Passenger:
    {sections['passenger']}
    
    ### Question:
    {sections['question']}
    
    ### Choices:
    {choices_text}
    
    Respond with ONLY the letter and full text of the correct answer.
    """.strip()
    
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": map_answer(text, answer_letter)
        }
    ]


def generate_full_prompt(text, answer):
    try:
        full_prompt = make_prompt(text, answer)

        prompt_str = tokenizer.apply_chat_template(
            full_prompt,
            tokenize=False,
            add_generation_prompt=False
        )
        
        tokenized = tokenizer(
            prompt_str,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
    
        input_ids = tokenized['input_ids'][0]
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized['attention_mask'][0],
            "labels": labels
        }
        
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None
    

