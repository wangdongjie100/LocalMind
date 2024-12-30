import gradio as gr
import time
import os
import json

from llm_logic import *

llm = None

# Function to fetch available models using the 'ollama' command
def get_ollama_models():
    try:
        # Execute the command to fetch model list
        raw_output = os.popen("ollama list").read()

        # Split the output into lines
        lines = raw_output.splitlines()

        # Skip the header line and extract model names
        models = [line.split()[0] for line in lines[1:] if line.strip()]
        return models
    except FileNotFoundError:
        return ["Error: 'ollama' command not found. Is Ollama installed?"]
    except json.JSONDecodeError:
        return ["Error: Unable to parse 'ollama list' output as JSON."]
    except Exception as e:
        return [f"Error: {str(e)}"]
    

def apply_settings(temp, tokens, model_name):
    global llm
    llm = initialize_llm(model_name, temp, tokens)
    return f"✅ Settings Applied!"

def reset_button_text(button):
    import time
    time.sleep(3)  # Wait for 3 seconds
    return button
    

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history: list):
    global llm
    response = answer_generation_using_message(llm,history)
    print(history)
    history.append({"role": "assistant", "content": ""})
    for character in response:
        history[-1]["content"] += character
        time.sleep(0.001)
        yield history


with gr.Blocks() as demo:
    gr.Markdown("# Local Mind Chatbot")
    # Model Settings
    gr.Markdown("### Model Setting")
    with gr.Row():
        temperature_slider = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature", scale=1
        )
        max_tokens_slider = gr.Slider(
            minimum=64, maximum=8196, value=1024, step=64, label="Max Tokens", scale=1
        )
        model_dropdown = gr.Dropdown(
            choices=get_ollama_models(),
            value=get_ollama_models()[0] if get_ollama_models() else "",
            label="Model",
            scale=1
        )

        apply_button = gr.Button("Apply Settings",scale=1)
        
    
    llm = initialize_llm(model_dropdown.value, temperature_slider.value, max_tokens_slider.value)


    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )

    bot_msg = chat_msg.then(
        bot,
        inputs=[chatbot],
        outputs=chatbot,
        api_name="bot_response",
    )

    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None, like_user_message=True)

    apply_button.click(apply_settings, 
                    [temperature_slider, max_tokens_slider, model_dropdown], 
                    [apply_button]).then(
        lambda: reset_button_text("Apply Settings"),
        None,
        [apply_button])

demo.launch()
