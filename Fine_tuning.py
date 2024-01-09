from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True).cuda()

# Directory containing the cleaned text files
output_directory = '/home/adam/BlendScriptAI/Text_Files'

# Load the dataset
def load_dataset(file_path):
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)

# Tokenize all text files in the directory
datasets = [load_dataset(os.path.join(output_directory, filename)) for filename in os.listdir(output_directory) if filename.endswith('.txt')]

# Data collator used for dynamic padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust epochs based on performance
    per_device_train_batch_size=1,  # Reduced batch size
    gradient_accumulation_steps=4,  # Using gradient accumulation
    fp16=True,  # Using 16-bit floating point precision
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
