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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.knowledge_graph import KnowledgeGraph
from src.chatbot import Chatbot
from src.ingestion import load_or_create_index

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

test_queries = [
    "O que é lógica proposicional segundo a apostila?",
    "Como a apostila define uma proposição?",
    "O que são conectivos lógicos e quais são apresentados no material?",
    "O que é uma tabela-verdade e para que ela é utilizada?",
    "Como a apostila define tautologia, contradição e contingência?"
]

ground_truths = [
    "Lógica proposicional é o ramo da lógica que estuda proposições e as relações entre elas por meio de conectivos lógicos.",
    "Proposição é toda sentença declarativa que pode ser classificada como verdadeira ou falsa, mas não ambas.",
    "Conectivos lógicos são operadores que conectam proposições, como negação (¬), conjunção (∧), disjunção (∨), condicional (→) e bicondicional (↔).",
    "Tabela-verdade é um método utilizado para determinar o valor lógico de proposições compostas a partir dos valores lógicos das proposições simples.",
    "Tautologia é uma proposição composta que é sempre verdadeira; contradição é sempre falsa; contingência é aquela que pode ser verdadeira ou falsa dependendo dos valores das proposições componentes."
]


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

        chat_result = chatbot.chat(query)
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
