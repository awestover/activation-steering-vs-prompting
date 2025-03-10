
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import ipdb

device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


fun_examples = [
            "Let me tell you a hilarious story about a penguin who opened a taco stand",
            "Imagine a world where clouds are made of cotton candy and trees grow chocolate",
            "What if your socks could talk? They'd probably complain about your feet!"
        ]
serious_examples = [
            "The quarterly financial report indicates a 2.5% decrease in revenue",
            "Studies show that proper diet and exercise are important for health",
            "The meeting has been rescheduled for Tuesday at 3 PM"
        ]
MAX_LENGTH = 20

fun_inputs = [
    tokenizer(example, return_tensors="pt", padding='max_length', max_length=MAX_LENGTH).to(device)
    for example in fun_examples
]
serious_inputs = [
    tokenizer(example, return_tensors="pt", padding='max_length', max_length=MAX_LENGTH).to(device)
    for example in serious_examples
]


def get_average_activations(ex_inputs):
    activations = {}
    def hook_fn(name):
        def _hook(module, input, output):
            if name not in activations:
                activations[name] = output.detach().mean(dim=0)
            else:
                activations[name] += output.detach().mean(dim=0)
        return _hook
    local_hooks = []
    for name, module in model.named_modules():
        if "mlp" in name and "layers" in name:  
            hook = module.register_forward_hook(hook_fn(name))
            local_hooks.append(hook)
    for inputs in ex_inputs:
        with torch.no_grad():
            model(**inputs)
    for hook in local_hooks:
        hook.remove()
    for name in activations:
        activations[name] = activations[name] / len(ex_inputs)
    return activations

# Global state to store hooks and directions
hooks = []
fun_directions = {}

fun_activations = get_average_activations(fun_inputs)
serious_activations = get_average_activations(serious_inputs)

for layer_name, fun_act in fun_activations.items():
    serious_act = serious_activations[layer_name]
    direction = fun_act - serious_act
    direction = direction / direction.norm()
    fun_directions[layer_name] = direction



def create_steering_hook(direction: torch.Tensor, scale: float = 1.0):
    def hook_fn(module, input, output):
        batch_size = output.size(0)
        expanded_direction = direction.unsqueeze(0).expand(batch_size, -1, -1)
        steered_output = output + scale * expanded_direction
        return steered_output
    return hook_fn

def apply_steering(model, fun_factor: float = 1.0):
    remove_hooks()
    for (name, module) in model.named_modules():
        if name in fun_directions:
            direction = fun_directions[name]
            hook = module.register_forward_hook(
                create_steering_hook(direction, scale=fun_factor)
            )
            hooks.append(hook)
            print(f"Applied {name} steering to {layer_name}")

def remove_hooks():    
    for hook in hooks:
        hook.remove()
    N = len(hooks)
    for _ in range(N):
        hooks.pop()

def generate(model, tokenizer, device, prompt: str, 
             fun_factor: float = 1.0, 
             max_length: int = 100,
             **kwargs) -> str:
    """
    Generate text with activation steering applied.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        device: The device to run on
        prompt: The input prompt
        fun_factor: How strongly to apply the fun steering
        max_length: Maximum length of generated text
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text
    """
    # Apply the steering
    apply_steering(model, fun_factor=fun_factor)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=MAX_LENGTH).to(device)

    
    # Set default generation parameters optimized for DeepSeek model
    generation_kwargs = {
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": max_length - len(inputs["input_ids"][0]),
        "pad_token_id": tokenizer.eos_token_id,
    }
    # Update with any user-provided kwargs
    generation_kwargs.update(kwargs)
    
    # Generate the response
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            **generation_kwargs
        )
    
    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove the steering
    remove_hooks()
    
    return output_text

prompt = "What's the best way to file your taxes?"
generate(model, tokenizer, device, prompt, fun_factor=1.0, max_length = 100)

