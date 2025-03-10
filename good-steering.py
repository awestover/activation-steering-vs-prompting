from transformer_lens import HookedTransformer

# load TransformerLens model
model = HookedTransformer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# define a steering vector
cache_name = f"blocks.{layer_id}.hook_resid_post"
_, cache = model.run_with_cache("Love")
act_love = cache[cache_name]
_, cache = model.run_with_cache("Hate")
act_hate = cache[cache_name]
steering_vec = act_love-act_hate

# define the activation steering funtion
def act_add(steering_vec):
    def hook(activation):
        return activation + steering_vec
    return hook

# generate text while steering
test_sentence = "I think dogs are "
model.add_hook(name=cache_name, hook=act_add(steering_vec))
print(model.generate(test_sentence, max_new_tokens=10))
print("-"*20)
model.reset_hooks()
model.add_hook(name=cache_name, hook=fun_factory(-steering_vec))
print(model.generate(test_sentence, max_new_tokens=10))
model.reset_hooks()
