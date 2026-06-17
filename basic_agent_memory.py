from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

# 1. Init Ollama model
print("Connecting Ollama...")
llm = OllamaLLM(
    model = "ministral-3:3b",
    base_url= "http://127.0.0.1:11434"
)

print("Ollama connected!")

# 2. Keep chat history
history_chat: list = []

def format_history():
    """Format the history of the chat."""
    if not history_chat:
        return "No previous history."

    text: str = "Chat history:\n"
    for msg in history_chat:
        text += f"{msg['role']}: {msg['content']}\n"

    return text

# 3. Template with context
template = """Você é um especialista em DevOps e infraestrutura de TI.
Responda de forma prática, técnica e educational.
Lembre-se do contexto da conversa anterior.

{history}

Pergunta atual: {{question}}

Resposta:"""

def create_prompt_with_history():
    """Create the prompt with the history of the chat."""
    history_formated = format_history()
    final_template = template.format(
        history=history_formated
    )

    return PromptTemplate(
        template=final_template,
        input_variables=["question"]
    )

# 4. Chain of memory
def ask_question(question: str) -> str:
    """Ask the DevOps agent the question."""

    # Add question to history
    history_chat.append(
        {
            "role": "Você",
            "content": question,
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
    )

    # Create a prompt with the current history
    prompt = create_prompt_with_history()

    # Create chain
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Create response
    response = chain.invoke(question)

    # Add response to history
    history_chat.append(
        {
            "role": "agent",
            "content": response,
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
    )

    return response

# 5. Main interface
def main():
    """Main function."""
    print("\n Devops Agent with Memory")
    print("=" * 50)
    print("Digite 'sair' para finalizar")
    print("Digite 'limpar' para resetar o historico\n")

    global history_chat

    while True:
        question = input("Você: ").strip()

        if question.lower() == "sair":
            print("Até logo!")
            break

        if question.lower() == "limpar":
            history_chat.clear()
            print("Historico limpo!\n")
            continue

        if not question:
            continue

        print("\nAgent: ", end="", flush=True)
        try:
            response = ask_question(question)
            print(response.strip())
        except Exception as e:
            print(f"Erro: {e}")
            print("Dica: Certifique-se que 'ollama serve' está rodando")

            # Remove question from history if there is an error.
            history_chat.pop()

        print()

if __name__ == "__main__":
    main()