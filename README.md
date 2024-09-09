---
library_name: transformers
tags: []
---


# Mental Health Therapy Chatbot

**Model Name**: `tanusrich/Mental_Health_Chatbot`  
**Model Type**: LLaMA-based model fine-tuned for Mental Health Therapy assistance

## Overview

The **Mental Health Therapy Chatbot** is a conversational AI model designed to provide empathetic, non-judgmental support to individuals seeking mental health guidance. This model has been fine-tuned using a carefully curated dataset to offer responses that are considerate, supportive, and structured to simulate therapy-like conversations.

It is ideal for use in mental health support applications where users can receive thoughtful and compassionate replies, especially on topics related to anxiety, loneliness, and general emotional well-being.

## Model Details

- **Base Model**: [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- **Fine-Tuned Dataset**: A specialized dataset curated for mental health conversations.
- **Purpose**: Provide empathetic, supportive responses for individuals seeking mental health assistance.
- **Language**: English

## Model Architecture

This model is based on **LLaMA-2-7b** architecture. It is a causal language model (CausalLM), which generates responses based on the input prompt by predicting the next word in the sequence.

### Model Capabilities

- Understands mental health queries and provides compassionate responses.
- Helps users navigate feelings such as loneliness, anxiety, and stress.
- Suggests coping strategies based on the context of the conversation.
- Can handle complex and emotionally charged topics with care.

## Usage

To use this model for generating mental health support responses, you can load it with the Hugging Face `transformers` library.

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "tanusrich/Mental_Health_Chatbot"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate a response
def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example interaction
user_input = "I'm feeling lonely and anxious. What can I do?"
response = generate_response(user_input)
print("Chatbot: ", response)
```

### Model Parameters

- **Max Tokens**: 200
- **Temperature**: 0.7 (controls the randomness of predictions)
- **Top-k**: 50 (filters to consider the top 50 tokens by probability)
- **Top-p**: 0.9 (nucleus sampling to filter based on cumulative probability)
- **Repetition Penalty**: 1.2 (penalizes repeated phrases or words)

### Example Queries

- "I'm feeling lonely and isolated. Can you help me?"
- "I often get anxious before going to work. What can I do to feel better?"
- "What are some coping strategies for dealing with stress?"

## Fine-Tuning

This model was fine-tuned using the **QLoRA (Quantized LoRA)** method, leveraging LoRA (Low-Rank Adaptation) layers to allow efficient fine-tuning on resource-constrained hardware.

### Fine-Tuning Configuration:

- **LoRA attention dimension (`lora_r`)**: 64
- **LoRA scaling factor (`lora_alpha`)**: 16
- **LoRA dropout**: 0.1
- **Precision**: 4-bit quantization
- **Optimizer**: Paged AdamW 32-bit

## Intended Use

This model is designed to assist in non-clinical settings where users might seek empathetic conversation and mental health support. It is not intended to replace professional mental health services or clinical diagnosis.

## Limitations

- **Non-clinical**: This model is not a substitute for professional therapy or mental health counseling. Users in need of medical attention should seek licensed professionals.
- **Emotionally complex conversations**: While the model attempts to provide helpful and compassionate responses, it might not always fully grasp the complexity of certain topics.
- **Bias**: Like any language model, responses might inadvertently reflect biases present in the training data. Usage in sensitive contexts should be approached cautiously.

## Ethical Considerations

Mental health is a sensitive area, and while this model attempts to provide thoughtful and supportive responses, it is essential to ensure that users understand it is not a replacement for professional help. Users should be encouraged to seek assistance from licensed professionals for serious mental health issues.

## Citation

If you use this model, please cite the LLaMA-2 model and the fine-tuning process as follows:

```plaintext
@article{LLaMA2,
  title={LLaMA 2: Open Foundation and Fine-Tuned Chat Models},
  author={Meta AI},
  year={2023}
}

@misc{tanusrich2024MentalHealthChatbot,
  author = {Tanusri},
  title = {Mental Health Therapy Chatbot},
  year = {2024},
  url = {https://huggingface.co/tanusrich/Mental_Health_Chatbot}
}
```

