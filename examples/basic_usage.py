"""
Basic usage example for PEFT-SingLoRA.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from peft_singlora import setup_singlora


def main():
    setup_singlora()
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],  # Target attention layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    text = "This is a test sentence for classification."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = peft_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        print(f"Predictions: {predictions}")

    # optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)
    # ... training loop ...

    # peft_model.save_pretrained("./singlora-adapter")

    # from peft import PeftModel
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # peft_model = PeftModel.from_pretrained(model, "./singlora-adapter")


if __name__ == "__main__":
    main()
