# %%
import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
from huggingface_hub import notebook_login
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# %%
notebook_login()

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = '40960MB'  # It's cleaner to not use an f-string here since it's a static value

    # Load model without specifying device_map initially
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        max_memory={i: max_memory for i in range(n_gpus)}  # Configure maximum memory per GPU
    )

    # Manual device mapping example
    if n_gpus > 1:
        # Distribute model layers manually across available GPUs
        device_map = {i: list(range(i * len(model.transformer.h) // n_gpus,
                                     (i + 1) * len(model.transformer.h) // n_gpus))
                      for i in range(n_gpus)}
        model.parallelize(device_map)
    else:
        # If there's only one GPU, use it directly
        model.to('cuda:0')

    # Load tokenizer and set special tokens as needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token  # Needed for LLaMA tokenizer

    return model, tokenizer


# %% [markdown]
# No good

# %%
def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# %% [markdown]
# 1st Dataset

# %%
from datasets import load_dataset

# Path to the local dataset file
local_dataset_path = r'C:\Users\spite\Documents\FT-Llama2-HR_Chatbot\Dataset_1_Narrative_HR_Time_Tracking_Dataset.json'

# Load the dataset from the local file
dataset = load_dataset('json', data_files={'train': local_dataset_path}, split='train')

# Now you can use 'dataset' as you would with any dataset loaded from the Hugging Face hub


# %% [markdown]
# 2nd Dataset

# %%
from datasets import load_dataset

# New instruction dataset
dolly_dataset = "databricks/databricks-dolly-15k"


dataset = load_dataset(dolly_dataset, split="train")

# %% [markdown]
# 3rd Dataset

# %%
print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

# %% [markdown]
# For Dataset = databricks-dolly-15k & custom/mine

# %%
def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """

    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['response']}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    
    sample["text"] = formatted_prompt

    return sample


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

# %% [markdown]
# BitsAndBytesConfig -

# %%
def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

# %% [markdown]
#  LoRa configuration:

# %%
def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    
    
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
        
    )
     

    return config

# %% [markdown]
# target modules

# %%
# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# %%
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

# %% [markdown]
# Train 
#  pre-process the dataset and load the model using the set configurations

# %%
model_name = "meta-llama/Llama-2-7b-hf" 

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config)



# %%
## Preprocess dataset
max_length = get_max_length(model)

seed=0

dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)

# %% [markdown]
# Loop Training - not working

# %%
import os
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, find_all_linear_names, create_peft_config, get_peft_model
from peft.utils import print_trainable_parameters

def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=50,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     
    do_train = True
    
    # Launch training
    print("Training...")
    
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)    
    
    ###
    
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()

def loop_train(model, tokenizer, dataset, output_base_dir, num_iterations=10):
    for i in range(1, num_iterations + 1):
        output_dir = f"{output_base_dir}/iteration_{i}"
        print(f"\nStarting iteration {i}/{num_iterations}...")
        train(model, tokenizer, dataset, output_dir)

# Example usage:
# Replace these with your actual model, tokenizer, and dataset
model = model
tokenizer = tokenizer
dataset = dataset

output_base_dir = "results/llama2"
loop_train(model, tokenizer, dataset, output_base_dir)


# %% [markdown]
# OG TRaining

# %%
def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=70,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     
    do_train = True
    
    # Launch training
    print("Training...")
    
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)    
    
    ###
    
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()
    
    
output_dir = "results/llama2/final_checkpoint"
train(model, tokenizer, dataset, output_dir)

# %% [markdown]
#  save to a new directory and associated tokenize

# %%

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto",offload_buffers=True, torch_dtype=torch.bfloat16, offload_folder='./offload_folder')

model = model.merge_and_unload()

output_merged_dir = "results/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)

# %%
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

output_merged_dir = "results/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)

try:
    model = AutoPeftModelForCausalLM.from_pretrained(
        "results/llama2/final_checkpoint",  # Make sure this path is correct
        torch_dtype=torch.bfloat16,
        offload_buffers=True,
        offload_folder='./offload_folder'
    )

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        device_map = {
            i: list(range(i * len(model.transformer.h) // n_gpus, (i + 1) * len(model.transformer.h) // n_gpus))
            for i in range(n_gpus)
        }
        model.parallelize(device_map)
    else:
        model.to('cuda:0')

    model = model.merge_and_unload()
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    tokenizer_path = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_merged_dir)

except RuntimeError as e:
    print("RuntimeError:", e)
    if "CUDA out of memory" in str(e):
        print("Trying to free up CUDA memory")
        torch.cuda.empty_cache()
        # Consider reducing batch size or using more GPUs
except Exception as e:
    print("An error occurred:", e)


# %%
print(model)

# %% [markdown]
# clears cache

# %%
import torch
torch.cuda.empty_cache()


