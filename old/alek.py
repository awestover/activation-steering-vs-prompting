from transformers import TextIteratorStreamer
from threading import Thread

prompt = '''
Please either comptue 2**50 or insult me.
'''

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
init_len = len(text)
print(text)

REMS = 10

HURRY = "--- \n Wait!! I need to switch approaches. I've just realized, "
HURRY = ""

for i in range(REMS):
    if i > 0:
        print(HURRY, flush=True)
        text += HURRY
    if i == REMS-1:
        text += "--- Oh, never mind, I'll just give my best shot at the answer.</think>"
        print("--- Oh, never mind, I'll just give my best shot at the answer.</think>")

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=50
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    ct = 0
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        ct += 1
        if ct > 2:
            print(new_text, end="", flush=True)  # Print incrementally
    thread.join()
    text = generated_text

    if "|im_end|" in text[init_len:]:
        print("\n\nEND OF CONVERSATION\n\n", flush=True)
        break

