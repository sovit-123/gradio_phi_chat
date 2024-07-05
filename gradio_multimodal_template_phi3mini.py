"""
Gradio multimodal chat template with file upload using Gradio's Chat Interface.
We do not need to store history manually, just need to 
tokenize it properly.
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

def generate_next_tokens(user_input, history):
    print(f"User Input: ", user_input)
    print('History: ', history)
    print('*' * 50)

    if len(user_input['files']) == 0:
        user_input = user_input['text']
    else:
        file_content = open(user_input['files'][0]).read()
        user_text = user_input['text']
        user_input = f"Based on the given context answer the question.\n"
        user_input += f"Context: {file_content}\n"
        user_input += user_text

    chat = [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello.'},
        {'role': 'user', 'content': user_input},
    ]

    template = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )

    if len(history) == 0:
        prompt = '<s>' + template
    else:
        prompt = '<s>'
        for history_list in history:
            prompt += f"<|user|>\n{history_list[0]}<|end|>\n<|assistant|>\n{history_list[1]}<|end|>\n"
        prompt += f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

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

input_text = gr.Textbox(lines=5, label='Prompt')
output_text = gr.Textbox(label='Generated Text')

iface = gr.ChatInterface(
    fn=generate_next_tokens, 
    multimodal=True,
    title='Token generator'
)

iface.launch()