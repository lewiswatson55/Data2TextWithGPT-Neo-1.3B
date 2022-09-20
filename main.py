import datasets
from transformers import AutoTokenizer, GPTNeoForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import pandas as pd
import numpy as np
import wandb

# wandb.init(project="Data2TextWithGPT-Neo-1.3B", entity="lewiswatson")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
datacollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Tokenizer max length: ", tokenizer.model_max_length)

# pause = input("Waiting for you to press enter...")

train_df = pd.read_csv("webNLG2020_train-v3.csv", index_col=[0])
train_df = train_df.iloc[:35000, :]
train_df = train_df.sample(frac=1)

eval_df = pd.read_csv("webNLG2020_eval-v3.csv", index_col=[0])
eval_df = eval_df.iloc[:10, :] # Capping at 10 as more results in a cuda memory error...
eval_df = eval_df.sample(frac=1)

# Tokenize the training dataset
tokenized_train_dataset = train_df.apply(
    lambda x: tokenizer(x['input_text'], x['target_text'], padding='max_length', truncation=True, max_length=400),
    axis=1)
print("Training dataset tokenized")

# Tokenize the evaluation dataset
tokenized_eval_dataset = eval_df.apply(
    lambda x: tokenizer(x['input_text'], x['target_text'], padding='max_length', truncation=True, max_length=400),
    axis=1)
print("Evaluation dataset tokenized")

batch_size = 1
num_of_batches = int(len(train_df) / batch_size)
num_of_epochs = 1

# Check if GPU is available - if not, get a GPU
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU: " + torch.cuda.get_device_name(0))
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


# Move the model to the device specified above - GPU!
# model.to(dev) # Cuda out of memory error if I do this but GPU is used anyway?


# Compute BLEU score for the evaluation dataset
def compute_metrics(eval_pred):
    with torch.no_grad():
        predictions, labels = eval_pred
        predictions = predictions.argmax(-1)
        # Compute BLEU score
        bleu = datasets.load_metric("sacrebleu")
        bleu = bleu.compute(predictions=predictions, references=labels)
        print("BLEU score: ", bleu)
        return bleu



TrainingArguments = TrainingArguments(
    report_to=["wandb"],
    output_dir="./results",  # output directory
    num_train_epochs=num_of_epochs,  # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    learning_rate=1e-4,  # learning rate
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",
    # push_to_hub=True,
    # push_to_hub_model_id="GPT-Neo-1.3B_finetuned_webNLG2020",

)

trainer = Trainer(
    model=model,
    args=TrainingArguments,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=datacollator,
    compute_metrics=compute_metrics,
)

trainer.train()
# trainer.push_to_hub("GPT-Neo-1.3B_finetuned_webNLG2020 - 1 epoch")
