"""
Script de avaliação RAGAS para o KE-RAG.

Executa as test_queries predefinidas de forma automática e retorna
as métricas: faithfulness, answer_relevancy, context_precision, context_recall.

Uso: python evaluate.py  (executar a partir da pasta knowledge-enhanced-rag/)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "benchmark-knowledge-enhanced-rag")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.knowledge_graph import KnowledgeGraph
from src.chatbot import Chatbot
from src.ingestion import load_or_create_index

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

test_queries = [
    "O que são conectivos lógicos e quais são os cinco conectivos apresentados na Apostila de Lógica de Programação?",
    "Como a Apostila de Lógica de Programação define tabela-verdade e qual princípio a fundamenta?",
    "Como funciona a pesquisa binária e qual é a sua complexidade de tempo segundo o livro Entendendo Algoritmos?",
    "Qual é a estratégia de dividir para conquistar utilizada pelo quicksort conforme descrito em Entendendo Algoritmos?",
    "Quais são os principais domínios de programação e suas linguagens associadas segundo Sebesta em Conceitos de Linguagens de Programação?",
    "Como Szwarcfiter define complexidade de pior caso de um algoritmo no livro Estruturas de Dados e Seus Algoritmos?",
    "Como o livro Fundamentos da Programação de Computadores de Ascencio descreve a função do computador e a necessidade de algoritmos?",
    "Como Forbellone define algoritmo e o conceito de sequenciação no livro Lógica de Programação?",
    "Segundo Nilo Menezes no livro Introdução à Programação com Python, o que realmente significa saber programar?",
    "Como Manzano define o termo algoritmo no livro Algoritmos: Lógica para Desenvolvimento de Programação de Computadores e qual é a sua origem etimológica?"
]

ground_truths = [
    "Conectivos lógicos são palavras ou frases usadas para formar novas proposições a partir de outras proposições. Os cinco conectivos apresentados são: negação (~), conjunção (∧), disjunção (∨), condicional (→) e bicondicional (↔).",
    "A tabela-verdade é um dispositivo que apresenta todos os possíveis valores lógicos de uma proposição composta, correspondentes a todas as atribuições de valores às proposições simples. Ela é fundamentada no Princípio do Terceiro Excluído, segundo o qual toda proposição é verdadeira ou falsa, e o valor lógico de uma proposição composta depende unicamente dos valores lógicos das proposições simples componentes.",
    "A pesquisa binária funciona eliminando metade dos elementos a cada iteração, comparando o elemento central com o valor buscado. Para uma lista de n elementos, requer no máximo log₂n verificações, resultando em complexidade O(log n), muito mais eficiente que a pesquisa simples O(n) para listas grandes.",
    "A estratégia de dividir para conquistar consiste em identificar o caso-base mais simples e dividir o problema até chegar a ele. O quicksort escolhe um elemento pivô, separa o array em dois subarrays com elementos menores e maiores que o pivô, e ordena recursivamente cada subarray. No caso médio, possui complexidade O(n log n).",
    "Sebesta identifica cinco domínios principais: aplicações científicas (Fortran, ALGOL), aplicações comerciais (COBOL), inteligência artificial (LISP, Prolog), programação de sistemas (C, C++) e programação para a Web. Cada domínio possui características distintas que motivaram o desenvolvimento de linguagens específicas.",
    "A complexidade de pior caso é definida como o número máximo de passos que um algoritmo executa considerando todas as entradas possíveis: max Ei ∈ E {ti}. Ela fornece um limite superior para o tempo de execução em qualquer situação e é a medida mais utilizada por garantir que o algoritmo nunca ultrapassará esse limite independentemente da entrada.",
    "Segundo Ascencio, o computador é uma máquina que recebe, manipula e armazena dados, mas não possui iniciativa, independência ou criatividade, precisando de instruções detalhadas. Sua finalidade principal é o processamento de dados: receber dados de entrada, realizar operações e gerar uma resposta de saída, o que requer algoritmos bem definidos.",
    "Forbellone define algoritmo como um conjunto de regras formais para a obtenção de um resultado ou da solução de um problema. A sequenciação é uma convenção que rege o fluxo de execução do algoritmo, determinando a ordem das ações de forma linear, de cima para baixo, assim como se lê um texto, estabelecendo um padrão de comportamento seguido por qualquer pessoa.",
    "Segundo Menezes, saber programar não é decorar comandos, parâmetros e nomes estranhos. Programar é saber utilizar uma linguagem de programação para resolver problemas, ou seja, saber expressar uma solução por meio de uma linguagem de programação. A sintaxe pode ser esquecida, mas quem realmente sabe programar tem pouca dificuldade ao aprender uma nova linguagem.",
    "Segundo Manzano, o termo algoritmo deriva do nome do matemático Muhammad ibn Musā al-Khwārizmī. Do ponto de vista computacional, algoritmo é entendido como regras formais, sequenciais e bem definidas a partir do entendimento lógico de um problema, com o objetivo de transformá-lo em um programa executável por um computador, onde dados de entrada são transformados em dados de saída."
]


@traceable(name="ke-rag-query", run_type="chain")
def ke_rag_traced(chatbot, query):
    return chatbot.chat(query)


def evaluate_ke_rag():
    print("Inicializando KE-RAG...")

    # Garante que o índice FAISS existe (cria a partir de ../docs/ se necessário)
    load_or_create_index()

    try:
        knowledge_graph = KnowledgeGraph()
    except Exception as e:
        print(f"Aviso: Knowledge Graph indisponivel ({e}). Continuando sem KG.")
        knowledge_graph = None

    chatbot = Chatbot(knowledge_graph=knowledge_graph)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )

    print("\nColetando respostas para avaliacao RAGAS...\n")
    ragas_data = []

    for i, query in enumerate(test_queries):
        print(f"  [{i+1}/{len(test_queries)}] {query}")

        retrieval = chatbot.retriever.retrieve(query)
        contexts = [doc.page_content for doc in retrieval["docs"]]

        chat_result = ke_rag_traced(chatbot, query)
        answer = chat_result["answer"]

        ragas_data.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truths[i]
        })

    dataset = Dataset.from_list(ragas_data)

    eval_llm = ChatOpenAI(
        model="llama3.3-70b-instruct",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_api_base="https://inference.do-ai.run/v1",
        temperature=0,
    )

    print("\nExecutando avaliacao RAGAS...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=embeddings,
    )

    print("\n=== RESULTADOS RAGAS ===")
    print(result)

    df = result.to_pandas()
    print("\nDetalhes por query:")
    print(df.to_string())

    return result


if __name__ == "__main__":
    evaluate_ke_rag()
