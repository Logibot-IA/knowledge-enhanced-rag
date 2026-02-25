"""
Módulo principal do Chatbot KE-RAG.

Implementa a lógica de conversação usando LangChain + Digital Ocean GenAI Platform (LLaMA 3.3 70B)
combinada com o retriever KE-RAG.
"""

import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage

from src.retriever import KERagRetriever
from src.knowledge_graph import KnowledgeGraph


# Prompt template em português, tom amigável para iniciantes
PROMPT_TEMPLATE = """Você é um professor assistente de Lógica de Programação, especialista em ajudar alunos iniciantes de Computação. Seu tom é amigável, paciente e didático.

Use as informações abaixo para responder à pergunta do aluno de forma clara e completa.

--- CONTEXTO DAS APOSTILAS ---
{contexto_docs}

--- FATOS DO KNOWLEDGE GRAPH ---
{kg_facts}

--- PRÉ-REQUISITOS DO CONCEITO ---
{prerequisites}

--- PRÓXIMOS CONCEITOS A ESTUDAR ---
{next_concepts}

Instruções:
- Responda sempre em português brasileiro
- Use exemplos simples e práticos quando possível
- Se o aluno não entender algo, sugira os pré-requisitos listados acima
- Ao final, sugira o próximo conceito a estudar se houver
- Se não encontrar informação nas apostilas, use seu conhecimento geral sobre o tema
- Seja encorajador e positivo

Pergunta do aluno: {pergunta}"""


class Chatbot:
    """
    Chatbot de Lógica de Programação baseado em KE-RAG.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph | None = None):
        """
        Inicializa o chatbot com o retriever KE-RAG e o modelo LLM.

        Args:
            knowledge_graph: Instância do KnowledgeGraph (opcional).
        """
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_api_key:
            raise ValueError(
                "Variável de ambiente OPENAI_API_KEY é obrigatória. "
                "Obtenha sua chave em cloud.digitalocean.com/gen-ai"
            )

        self.llm = ChatOpenAI(
            model="llama3.3-70b-instruct",
            openai_api_key=openai_api_key,
            openai_api_base="https://inference.do-ai.run/v1",
            temperature=0.7,
            max_tokens=1024,
        )

        self.retriever = KERagRetriever(knowledge_graph=knowledge_graph)

        # Histórico de conversas por sessão (últimas 5 mensagens)
        self.memorias: dict[str, ConversationBufferWindowMemory] = {}

    def _obter_memoria(self, session_id: str) -> ConversationBufferWindowMemory:
        """
        Obtém ou cria a memória de conversa para uma sessão.

        Args:
            session_id: Identificador da sessão.

        Returns:
            Memória de conversa da sessão.
        """
        if session_id not in self.memorias:
            self.memorias[session_id] = ConversationBufferWindowMemory(
                k=5,
                return_messages=True,
                memory_key="historico",
            )
        return self.memorias[session_id]

    def chat(self, pergunta: str, session_id: str = "default") -> dict:
        """
        Processa uma pergunta e retorna a resposta do chatbot.

        Args:
            pergunta: Pergunta do usuário.
            session_id: Identificador da sessão de conversa.

        Returns:
            Dicionário com:
                - answer: Resposta gerada pelo chatbot
                - prerequisites: Pré-requisitos do conceito identificado
                - next_concepts: Próximos conceitos sugeridos
        """
        # Recupera contexto via KE-RAG
        resultado = self.retriever.retrieve(pergunta)

        docs = resultado["docs"]
        kg_facts = resultado["kg_facts"]
        prerequisites = resultado["prerequisites"]
        next_concepts = resultado["next_concepts"]

        # Formata o contexto dos documentos
        if docs:
            contexto_docs = "\n\n".join([
                f"[Fonte: {doc.metadata.get('fonte', 'desconhecida')}, "
                f"Pág. {doc.metadata.get('pagina', '?')}]\n{doc.page_content}"
                for doc in docs
            ])
        else:
            contexto_docs = "Nenhum trecho relevante encontrado nas apostilas."

        # Formata pré-requisitos e próximos conceitos
        prereqs_texto = (
            ", ".join(prerequisites) if prerequisites
            else "Nenhum pré-requisito específico identificado."
        )
        proximos_texto = (
            ", ".join(next_concepts) if next_concepts
            else "Continue praticando o conceito atual."
        )

        # Monta o prompt
        prompt_usuario = PROMPT_TEMPLATE.format(
            contexto_docs=contexto_docs,
            kg_facts=kg_facts or "Nenhum fato do Knowledge Graph disponível.",
            prerequisites=prereqs_texto,
            next_concepts=proximos_texto,
            pergunta=pergunta,
        )

        # Obtém o histórico da sessão
        memoria = self._obter_memoria(session_id)
        historico = memoria.load_memory_variables({}).get("historico", [])

        # Monta as mensagens com histórico
        mensagens = historico + [HumanMessage(content=prompt_usuario)]

        # Chama o LLM
        resposta = self.llm.invoke(mensagens)
        resposta_texto = resposta.content

        # Salva no histórico
        memoria.save_context(
            {"input": pergunta},
            {"output": resposta_texto},
        )

        return {
            "answer": resposta_texto,
            "prerequisites": prerequisites,
            "next_concepts": next_concepts,
        }
