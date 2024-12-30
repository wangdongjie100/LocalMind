from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage



def initialize_llm(model_name="llama3.2", temperature=0.7, max_tokens=150):
    # Initialize the Ollama model
    llm = OllamaLLM(model=model_name, temperature=temperature, max_tokens=max_tokens)
    return llm

def answer_generation_using_message(llm, message):
    prompt = """You are an helpful assistant for question-answering tasks. 
        Now, review the user question: {question}
        Think carefully about the question. Provide an accurate answer to this question. 
        Use concise, clear and easy to understand way to write the sentences.
        Answer: """
    prompt_formatted = prompt.format(question=message)
    generation = llm.invoke([HumanMessage(content=prompt_formatted)])
    return generation

def answer_generation_using_file_and_message(llm, context, message):
    prompt = """You are an assistant for question-answering tasks. 
            Here is the context to use to answer the question:
            {context} 
            Think carefully about the above context. 
            Now, review the user question:
            {question}
            Provide an answer to this questions using only the above context. 
            Keep the answer concise and easy to understand.
            Answer:"""
    prompt_formatted = prompt.format(context=context,question=message)
    generation = llm.invoke([HumanMessage(content=prompt_formatted)])
    return generation

