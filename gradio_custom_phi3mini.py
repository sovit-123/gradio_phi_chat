"""
Gradio chat template using custom template. We store and manage history 
on our own.
"""

import gradio as gr
import threading

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer,
    BitsAndBytesConfig
)

device = 'cuda'

quant_config = BitsAndBytesConfig(
    load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Phi-3-mini-4k-instruct',
    quantization_config=quant_config,
    device_map=device
)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

CONTEXT_LENGTH = 3800 # This uses around 9.9GB of GPU memory when highest context length is reached.

history = ''

def generate_next_tokens(user_input):
    global history

    print('History: ', history)
    print('*' * 50)

    chat = [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello.'},
        {'role': 'user', 'content': user_input},
    ]

    template = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=False
    )

    # prompt =  history + template if len(history) > 1 else template
    prompt =  history + user_input + '<|end|>\n<|assistant|>\n' if len(history) > 1 else template

    print('Prompt: ', prompt)
    print('*' * 50)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

    # A way to manage context length + memory for best results.
    print('Global context length till now: ', input_ids.shape[1])
    if input_ids.shape[1] > CONTEXT_LENGTH:
        input_ids = input_ids[:, -CONTEXT_LENGTH:]
        attention_mask = attention_mask[:, -CONTEXT_LENGTH:]

    print('-' * 100)

    generate_kwargs = dict(
        {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)},
        streamer=streamer,
        max_new_tokens=1024,
    )

    thread = threading.Thread(
        target=model.generate, 
        kwargs=generate_kwargs
    )

    thread.start()

    outputs = []
    for new_token in streamer:
        outputs.append(new_token)
        final_output = ''.join(outputs)

        yield final_output
    
    if len(history) > 1:
        history += f"{user_input}<|end|>\n<|assistant|>\n{final_output}<|end|>\n<|user|>\n"
    else:
        history = f"{template}{final_output}<|end|>\n<|user|>\n"

input_text = gr.Textbox(lines=5, label='Prompt')
output_text = gr.Textbox(label='Generated Text')

iface = gr.Interface(
    fn=generate_next_tokens, 
    inputs=input_text, 
    outputs=output_text, 
    title='Token generator'
)

iface.launch()