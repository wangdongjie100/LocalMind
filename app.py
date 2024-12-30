import gradio as gr
import os
import json
from llm_logic import initialize_llm, answer_generation_using_message, answer_generation_using_file_and_message
from utils import file_load,retriver_prepare

# Global state
chat_history = []
model_settings = {"temperature": 0.7, "max_tokens": 100, "model": "qwen2.5"}
llm = initialize_llm(model_settings["model"], model_settings["temperature"], model_settings["max_tokens"])
retriver = None

# Retrieve the list of Ollama models
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

# Chat functionality
def chatbot(input_text):
    global chat_history, retriver
    if retriver != None:
        docs = retriver.invoke(input_text)
        context = docs[0].page_content
        reply = answer_generation_using_file_and_message(llm,context,input_text)
        chat_history.append(("User", input_text))
        chat_history.append(("ChatBot", reply))
    else:
        reply = answer_generation_using_message(llm,input_text)
        chat_history.append(("User", input_text))
        chat_history.append(("ChatBot", reply))
    return chat_history, ""

# Update model settings
def update_settings(temperature, max_tokens, model_name):
    model_settings["temperature"] = temperature
    model_settings["max_tokens"] = max_tokens
    model_settings["model"] = model_name
    global llm 
    llm = initialize_llm(model_settings["model"], model_settings["temperature"], model_settings["max_tokens"])
    return f"Settings Updated: Temp={temperature}, MaxTokens={max_tokens}, Model={model_name}"

# Handle file uploads
def handle_file(file):
    contents = file_load(file)
    global retriver
    retriver = retriver_prepare(contents,k=2)
    return f"File '{file.name}' uploaded successfully!"

def create_gradio_interface():
    # Build Gradio interface
    with gr.Blocks (css="""
                    .equal-height {
                        display: flex;
                        align-items: stretch; /* Ensure equal height for both columns */
                    }

                    .column {
                        flex: 1; /* Allow columns to stretch in width */
                        display: flex;
                        flex-direction: column;
                    }

                    .chat-container {
                        display: flex;
                        flex-direction: column;
                        flex: 1;
                        height: 100%;
                        max-height: 600px; /* Limit overall height */
                    }

                    .chatbox {
                        flex-grow: 1; /* Fill remaining space in parent container */
                        overflow-y: auto; /* Add scroll support */
                        border: 1px solid #ccc;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    """) as app:
        gr.Markdown("# Chat Application")

        # Layout with two columns
        with gr.Row(elem_classes="equal-height"):  # Use the equal-height class to ensure equal height for both columns
            # Left column: Chat functionality
            with gr.Column(elem_classes="column", scale=3):  # Left column occupies a larger proportion
                gr.Markdown("### Chat Interface")
                chat_box = gr.Chatbot(elem_classes="chatbox")  # Chat history box
                user_input = gr.Textbox(label="Your message")
                send_button = gr.Button("Send")
                # Trigger chat functionality with button click or Enter key
                send_button.click(
                    chatbot,
                    inputs=[user_input],
                    outputs=[chat_box, user_input],  # Update chatbox and clear input box simultaneously
                )
                user_input.submit(
                    chatbot,
                    inputs=[user_input],
                    outputs=[chat_box, user_input],  # Support sending message with Enter key
                )

            # Right column: Settings and file upload
            with gr.Column(elem_classes="column", scale=1):  # Right column occupies a smaller proportion
                gr.Markdown("### Settings")
                temperature = gr.Slider(0, 1, step=0.1, value=model_settings['temperature'], label="Temperature")
                max_tokens = gr.Slider(50, 200, step=10, value=model_settings['max_tokens'], label="Max Tokens")
                model_dropdown = gr.Dropdown(
                    choices=get_ollama_models() or ["No models available"],  # Dynamically fetch model list
                    label="Select Model",
                    value=get_ollama_models()[0],  # Default selected model
                )
                update_button = gr.Button("Update Settings")
                status = gr.Textbox(label="Status", interactive=False)

                # Update settings button
                update_button.click(
                    update_settings,
                    inputs=[temperature, max_tokens, model_dropdown],
                    outputs=status,
                )

                gr.Markdown("### File Upload")
                file_upload = gr.File(label="Upload your file")
                file_status = gr.Textbox(label="Upload Status", interactive=False)
                file_upload.upload(handle_file, inputs=file_upload, outputs=file_status)
    return app 

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()
