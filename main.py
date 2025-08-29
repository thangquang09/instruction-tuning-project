import torch
from peft import PeftModel
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from load_data import load_dataset, prepare_model_data
from load_model import get_model


class LogLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"Step {state.global_step} - Loss: {logs['loss']:.4f}")


dataset = load_dataset()
train_dataset = prepare_model_data(dataset["train_dataset"])
valid_dataset = prepare_model_data(dataset["valid_dataset"])
tokenizer, model = get_model()

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir="llama3-8b-sat-reading/",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=True,
    report_to="none",
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=training_args,
    data_collator=data_collator,
    callbacks=[LogLossCallback()],
)

model.config.use_cache = False
model.enable_input_require_grads()
model = torch.compile(model)

trainer.train()

model.save_pretrained("trained-model")
PEFT_MODEL = "thangquang09/InstructionTuning-llama3-8b-sat-reading-solver"

model.push_to_hub(PEFT_MODEL, use_auth_token=True)