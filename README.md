# AI News Generator with GPT-2

**AI News Generator with GPT-2** is a project that leverages the GPT-2 language model to generate coherent and contextually relevant news articles. This project demonstrates how fine-tuning a pre-trained language model can be used to create high-quality, human-like news content.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [License](#license)
- [Contact](#contact)

## Introduction

This project uses OpenAI's GPT-2 model to generate news articles based on input prompts. By fine-tuning GPT-2 on a news dataset, the model learns to produce text that resembles actual news content. The result is an AI-powered news generator capable of creating diverse and engaging news stories.

## Features

- Fine-tuned GPT-2 model for news text generation
- Flexible input prompts for generating various types of news content
- GPU acceleration for faster model training and inference
- Easy-to-use interface for generating and evaluating news articles

## Requirements

To run this project, you need:

- Python 3.6 or higher
- PyTorch 1.7.0 or higher
- Transformers library from Hugging Face
- Other Python packages as listed in `requirements.txt`

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/ai-news-generator-gpt2.git
    cd ai-news-generator-gpt2
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download pre-trained GPT-2 model:**

    The script will automatically handle this when you run the code.

## Usage

To generate news articles using the pre-trained GPT-2 model, you can run the following script:

```python
# Example usage script
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example prompt
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
attention_mask = torch.ones(input_ids.shape, device=device)

# Generate text
output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)


Training
To fine-tune the GPT-2 model on your own news dataset, follow these steps:

Prepare your dataset in text format, with each news article separated by a new line.

Update the train_inputs and train_targets in the NewsDataset class with your data.

Run the training script:
# Training script
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch
from torch.utils.data import DataLoader

# Initialize model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create dataset and dataloader
train_dataset = NewsDataset(train_inputs, train_targets, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Set optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
for epoch in range(3):  # Adjust epochs as needed
    train_loss = train_epoch(model, train_loader, optimizer)
    print(f'Epoch {epoch + 1}, Loss: {train_loss}')


Contact
For any questions or feedback, please contact:

Email: vlad0067vlad@gmail.com
GitHub: vladstelmakh
