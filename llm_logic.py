from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def initialize_llm_chain(model_name="llama3.2", temperature=0.7, max_tokens=150):
    """
    Initialize LangChain's LLM Chain based on the Ollama model.

    Parameters:
    - model_name (str): The model name, e.g., "llama2".
    - temperature (float): The randomness of the model's generation, range [0, 1].
    - max_tokens (int): The maximum number of output tokens.

    Returns:
    - LLMChain instance.
    """
    # Initialize the Ollama model
    llm = OllamaLLM(model=model_name, temperature=temperature, max_tokens=max_tokens)
    
    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer the question: {question}"
    )
    
    # Create the LLM Chain
    chain = prompt | llm
    return chain

def ask_question_with_chain(chain, question):
    """
    Use the initialized LLM Chain to answer a question.

    Parameters:
    - chain (LLMChain): The initialized LLMChain instance.
    - question (str): The user's question.

    Returns:
    - str: The model's response.
    """
    response = chain.invoke({"question":question})
    return response

# Example: Using the utility functions
if __name__ == "__main__":
    # Initialize the LLM Chain
    model_settings = {
        "model_name": "llama3.2",
        "temperature": 0.7,
        "max_tokens": 150
    }
    chain = initialize_llm_chain(**model_settings)

    # Ask a question
    question = "What is the capital of France?"
    answer = ask_question_with_chain(chain, question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
