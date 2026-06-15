from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Init Ollama model
print("Connecting Ollama...")
llm = OllamaLLM(
    model = "ministral-3:3b",
    base_url= "http://127.0.0.1:11434"
)

# 2. Create prompt template Devops
template = """
Você é um especialista em DevOps e infraestrutura de TI.
Responda a pergunta de forma clara, prática e técnica.
Se a pergunta não for sobre DevOps/TI, mencione que você é especializado nessa área.
 
Pergunta: {pergunta}
 
Resposta:
"""

prompt = PromptTemplate(
    template = template,
    input_variables=["pergunta"]
)

# 3. Create a chain with LCEL sintaxe
chain = (
    {"pergunta": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)


# 4. Function to start interaction
def ask_a_question(question: str) -> str:
    """Ask a question to Devops agent."""
    answer = chain.invoke(question)
    return answer


# 5. Interaction loop
if __name__ == "__main__":
    print("Conectado ao Ollama com sucesso!")
    print("\n Devops agent - Ollama")
    print("=" * 50)
    print("Digite 'sair' para finalizar\n")

    while True:
        question = input("Você: ").strip()

        if question.lower() == "sair":
            print("Até logo!")
            break

        if not question:
            continue

        print("\nAgente: ", end=" ", flush=True)
        try:
            answer = ask_a_question(question)
            print(answer)
        except Exception as e:
            print(f"Erro: {e}")
            print("Dica: Certifique-se que 'Ollama server' está rodando.")
            print()
