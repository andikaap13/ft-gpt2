from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

device = "cpu"
trained_model_name = 'ft-gpt2'
train_file_path = 'data.txt'

# Load Pretrained Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')


# Original Prompt
prompt = "Tertawa adalah sebuah tindakan yang membuat hati bahagia"


# Inference Pretrained Model
tokenized_input = tokenizer(prompt, return_tensors="pt").to(device)

generated_ids = model.generate(
    tokenized_input.input_ids,
    max_new_tokens=512
)
print(tokenizer.decode(generated_ids[0]))


def load_dataset(file_path, tokenizer, block_size = 512):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

def train(train_file_path,
            model,
            tokenizer,
            output_dir,
            overwrite_output_dir,
            per_device_train_batch_size,
            num_train_epochs):
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
    )
      
    trainer.train()
    trainer.save_model()

train(
    train_file_path=train_file_path,
    model=model,
    tokenizer=tokenizer,
    output_dir=trained_model_name,
    overwrite_output_dir=False,
    per_device_train_batch_size=8,
    num_train_epochs=5
)


# Load Trained Model
tokenizer = GPT2Tokenizer.from_pretrained(trained_model_name)
model = GPT2LMHeadModel.from_pretrained(trained_model_name)


# Inference Trained Model
model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
print(tokenizer.decode(generated_ids[0]))