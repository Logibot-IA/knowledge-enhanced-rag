"""
Módulo de ingestão de PDFs para o KE-RAG.

Responsável por:
- Carregar PDFs da pasta data/apostilas/
- Dividir em chunks de texto
- Gerar embeddings e salvar no índice FAISS
"""

import os
from pathlib import Path

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Diretórios padrão
BASE_DIR = Path(__file__).resolve().parent.parent
APOSTILAS_DIR = BASE_DIR / "data" / "apostilas"
FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss_index"

# Modelo de embeddings gratuito (não requer API key)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def carregar_pdfs(pasta: Path = APOSTILAS_DIR) -> list[dict]:
    """
    Carrega todos os PDFs de uma pasta e extrai o texto de cada página.

    Args:
        pasta: Caminho para a pasta com os PDFs.

    Returns:
        Lista de dicionários com 'texto' e 'fonte' (nome do arquivo).
    """
    documentos = []
    arquivos_pdf = list(pasta.glob("*.pdf"))

    if not arquivos_pdf:
        print(f"Aviso: Nenhum PDF encontrado em '{pasta}'.")
        return documentos

    for arquivo in arquivos_pdf:
        print(f"Carregando: {arquivo.name}")
        try:
            with fitz.open(str(arquivo)) as doc:
                for num_pagina, pagina in enumerate(doc, start=1):
                    texto = pagina.get_text().strip()
                    if texto:
                        documentos.append({
                            "texto": texto,
                            "fonte": arquivo.name,
                            "pagina": num_pagina,
                        })
        except Exception as e:
            print(f"Erro ao carregar '{arquivo.name}': {e}")

    print(f"Total de páginas carregadas: {len(documentos)}")
    return documentos


def dividir_em_chunks(documentos: list[dict]) -> list:
    """
    Divide os documentos em chunks menores para indexação.

    Args:
        documentos: Lista de documentos com texto e metadados.

    Returns:
        Lista de objetos Document do LangChain.
    """
    from langchain.schema import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = []
    for doc in documentos:
        partes = splitter.split_text(doc["texto"])
        for parte in partes:
            chunks.append(Document(
                page_content=parte,
                metadata={"fonte": doc["fonte"], "pagina": doc["pagina"]},
            ))

    print(f"Total de chunks gerados: {len(chunks)}")
    return chunks


def obter_embeddings() -> HuggingFaceEmbeddings:
    """
    Inicializa o modelo de embeddings.

    Returns:
        Instância de HuggingFaceEmbeddings.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def criar_indice(chunks: list) -> FAISS:
    """
    Cria um novo índice FAISS a partir dos chunks.

    Args:
        chunks: Lista de Documents do LangChain.

    Returns:
        Índice FAISS criado.
    """
    print("Gerando embeddings e criando índice FAISS...")
    embeddings = obter_embeddings()
    indice = FAISS.from_documents(chunks, embeddings)
    indice.save_local(str(FAISS_INDEX_DIR))
    print(f"Índice FAISS salvo em '{FAISS_INDEX_DIR}'.")
    return indice


def carregar_indice() -> FAISS:
    """
    Carrega o índice FAISS existente do disco.

    Returns:
        Índice FAISS carregado.
    """
    embeddings = obter_embeddings()
    indice = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("Índice FAISS carregado do disco.")
    return indice


def load_or_create_index() -> FAISS:
    """
    Verifica se o índice FAISS já existe e o carrega; caso contrário, cria um novo.

    Returns:
        Índice FAISS pronto para uso.

    Raises:
        RuntimeError: Se não houver PDFs e o índice não existir.
    """
    indice_existe = (FAISS_INDEX_DIR / "index.faiss").exists()

    if indice_existe:
        print("Índice FAISS encontrado. Carregando...")
        return carregar_indice()

    print("Índice FAISS não encontrado. Criando novo índice...")
    documentos = carregar_pdfs()

    if not documentos:
        raise RuntimeError(
            "Nenhum PDF encontrado e nenhum índice existente. "
            f"Adicione PDFs em '{APOSTILAS_DIR}' e tente novamente."
        )

    chunks = dividir_em_chunks(documentos)
    return criar_indice(chunks)


def reindexar() -> FAISS:
    """
    Força a recriação do índice FAISS a partir dos PDFs atuais.

    Returns:
        Novo índice FAISS.

    Raises:
        RuntimeError: Se não houver PDFs disponíveis.
    """
    print("Reindexando PDFs...")
    documentos = carregar_pdfs()

    if not documentos:
        raise RuntimeError(
            f"Nenhum PDF encontrado em '{APOSTILAS_DIR}'. "
            "Adicione PDFs antes de reindexar."
        )

    chunks = dividir_em_chunks(documentos)
    return criar_indice(chunks)
