"""
Script de avaliação RAGAS para o KE-RAG.

Executa as test_queries predefinidas de forma automática e retorna
as métricas: faithfulness, answer_relevancy, context_precision, context_recall.

Uso: python main.py  (executar a partir da pasta knowledge-enhanced-rag/)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_settings import (
    build_embeddings,
    build_ragas_llm,
    configure_environment,
    finish_usage_tracker,
    run_ragas,
    salvar,
    start_usage_tracker,
)

configure_environment("benchmark-knowledge-enhanced-rag")

from src.knowledge_graph import KnowledgeGraph
from src.chatbot import Chatbot
from src.ingestion import load_or_create_index

from langsmith import traceable

test_queries = [
    # FÁCEIS
    "O que significa ‘lógica de programação’ em palavras simples?",
    "De um jeito bem direto: o que é um algoritmo?",
    "Qual é a diferença entre constante e variável?",
    "Pra que serve o comando ‘leia’ em um algoritmo?",

    # MÉDIAS
    "O que é um comando de atribuição e por que o tipo do dado precisa ser compatível com o tipo da variável?",
    "O que são operadores aritméticos (como +, -, * e /) e pra que eles servem?",
    "Pra que servem os operadores relacionais numa expressão?",

    # DIFÍCEIS
    "O que é uma ‘expressão lógica’?",
    "Em uma repetição, o que é um contador e como ele é incrementado?",
    "Como funciona a repetição ‘repita ... até’ e o que ela garante sobre a execução do bloco?"
]


ground_truths = [
    # FÁCEIS
    "Lógica de programação é o uso correto das leis do pensamento, da ‘ordem da razão’ e de processos formais de raciocínio e simbolização na programação de computadores, com o objetivo de produzir soluções logicamente válidas e coerentes para resolver problemas.",
    "Um algoritmo é uma sequência de passos bem definidos que têm por objetivo solucionar um determinado problema.",
    "Um dado é constante quando não sofre variação durante a execução do algoritmo: seu valor permanece constante do início ao fim (e também em execuções diferentes ao longo do tempo). Já um dado é variável quando pode ser alterado em algum instante durante a execução do algoritmo, ou quando seu valor depende da execução em um certo momento ou circunstância.",
    "O comando de entrada de dados ‘leia’ é usado para que o algoritmo receba os dados de que precisa: ele tem a finalidade de atribuir o dado fornecido à variável identificada, seguindo a sintaxe leia(identificador) (por exemplo, leia(X) ou leia(A, XPTO, NOTA)).",

    # MÉDIAS
    "Um comando de atribuição permite fornecer um valor a uma variável. O tipo do dado atribuído deve ser compatível com o tipo da variável: por exemplo, só se pode atribuir um valor lógico a uma variável declarada como do tipo lógico.",
    "Operadores aritméticos são o conjunto de símbolos que representam as operações básicas da matemática (por exemplo: + para adição, - para subtração, * para multiplicação e / para divisão). Para potenciação e radiciação, o livro indica o uso das palavras-chave pot e rad.",
    "Operadores relacionais são usados para realizar comparações entre dois valores de mesmo tipo primitivo. Esses valores podem ser constantes, variáveis ou expressões aritméticas, e esses operadores são comuns na construção de equações.",

    # DIFÍCEIS
    "Uma expressão lógica é aquela cujos operadores são lógicos ou relacionais e cujos operandos são relações, variáveis ou constantes do tipo lógico.",
    "Um contador é um modo de contagem feito com a ajuda de uma variável com um valor inicial, que é incrementada a cada repetição. Incrementar significa somar um valor constante (normalmente 1) a cada repetição.",
    "A estrutura de repetição ‘repita ... até’ permite que um bloco (ou ação primitiva) seja repetido até que uma determinada condição seja verdadeira. Pela sintaxe da estrutura, o bloco é executado pelo menos uma vez, independentemente da validade inicial da condição."
]


@traceable(name="ke-rag-query", run_type="chain")
def ke_rag_traced(chatbot, query, callbacks=None):
    return chatbot.chat(query, callbacks=callbacks)


def evaluate_ke_rag():
    print("Inicializando KE-RAG...")

    load_or_create_index()

    try:
        knowledge_graph = KnowledgeGraph()
    except Exception as e:
        print(f"Aviso: Knowledge Graph indisponivel ({e}). Continuando sem KG.")
        knowledge_graph = None

    chatbot = Chatbot(knowledge_graph=knowledge_graph)

    embeddings = build_embeddings()

    for run in range(5):
        print(f"\n=== RODADA {run + 1}/5 ===")
        eval_llm = build_ragas_llm()

        print("Coletando respostas para avaliacao RAGAS...")
        ragas_data = []

        for i, query in enumerate(test_queries):
            print(f"  [{i + 1}/{len(test_queries)}] {query}")

            retrieval = chatbot.retriever.retrieve(query)
            contexts = [doc.page_content for doc in retrieval["docs"]]

            tracker, started_at = start_usage_tracker()
            chat_result = ke_rag_traced(chatbot, query, callbacks=[tracker])
            answer = chat_result["answer"]

            ragas_item = {
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truths[i]
            }
            ragas_item.update(finish_usage_tracker(tracker, started_at))
            ragas_data.append(ragas_item)

        df_resultado = run_ragas(ragas_data, eval_llm, embeddings)
        salvar(df_resultado, nome_base=f"knowledge-enhanced-rag-run-{run + 1}")


if __name__ == "__main__":
    evaluate_ke_rag()

