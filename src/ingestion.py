"""
Módulo de ingestão de PDFs para o KE-RAG.

Responsável por:
- Carregar PDFs da pasta data/apostilas/
- Dividir em chunks de texto
- Gerar embeddings e salvar no índice FAISS
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Diretórios padrão
BASE_DIR = Path(__file__).resolve().parent.parent
APOSTILAS_DIR = BASE_DIR / "data" / "apostilas"
FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss_index"

# Modelo de embeddings (igual ao hybrid-rag)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def carregar_pdfs(pasta: Path = APOSTILAS_DIR) -> list:
    """
    Carrega todos os PDFs de uma pasta usando PyPDFDirectoryLoader.

    Args:
        pasta: Caminho para a pasta com os PDFs.

    Returns:
        Lista de objetos Document do LangChain.
    """
    arquivos_pdf = list(pasta.glob("*.pdf"))

    if not arquivos_pdf:
        print(f"Aviso: Nenhum PDF encontrado em '{pasta}'.")
        return []

    loader = PyPDFDirectoryLoader(str(pasta))
    documentos = loader.load()
    print(f"Total de páginas carregadas: {len(documentos)}")
    return documentos


def dividir_em_chunks(documentos: list) -> list:
    """
    Divide os documentos em chunks menores para indexação.

    Args:
        documentos: Lista de objetos Document do LangChain.

    Returns:
        Lista de objetos Document do LangChain divididos em chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documentos)
    print(f"Total de chunks gerados: {len(chunks)}")
    return chunks


def obter_embeddings() -> HuggingFaceEmbeddings:
    """
    Inicializa o modelo de embeddings.

    Returns:
        Instância de HuggingFaceEmbeddings.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


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
