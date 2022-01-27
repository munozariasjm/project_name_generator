from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "./trained_model"
model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained("t5-base")


class NameGenerator:
    def __init__(self):
        self.tokenizer = tokenizer

    def generate(self, description: str) -> str:
        description = str(description.lower().replace("\n", ""))
        with torch.no_grad():
            tokenized_text = tokenizer(
                description, truncation=True, padding=True, return_tensors="pt"
            )
            source_ids = tokenized_text["input_ids"]  # torch.from_numpy(
            # .to(device, dtype=torch.long)
            source_mask = tokenized_text["attention_mask"]
            generated_ids = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=150,
                num_beams=5,
                repetition_penalty=1,
                length_penalty=1,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        return tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
