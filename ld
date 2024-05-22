import openai
from typing import Iterator, List

from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# Определяем ваш кастомный загрузчик документов
class CustomDocumentLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line.strip(),
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1

# Инициализируем загрузчик и собираем все документы в список
loader = CustomDocumentLoader(file_path='Documents/tech.txt')
documents = list(loader.lazy_load())

# Инициализируем разделитель текста
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=70,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([doc.page_content for doc in documents])


# Инициализируем локальное хранилище для кеша
store = LocalFileStore("cache/")

# Создаем кастомный класс для эмбеддингов
class CustomAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str, api_base_url: str):
        openai.api_key = api_key
        openai.api_base = api_base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)

    def embed(self, text: str) -> List[float]:
        response = openai.Completion.create(
            model="text-embedding-ada-002",  # или другая модель, поддерживающая эмбеддинги
            prompt=text,
            max_tokens=512
        )
        return response['choices'][0]['embedding']

openai_api_key = "sk-OUM4cuOopHmkLFGB61F15375A72042B7AdDaB0D98bC4C7D0"
custom_api_base_url = "https://apisbost.top/v1/chat"

underlying_embeddings = CustomAPIEmbeddings(
    api_key=openai_api_key,
    api_base_url=custom_api_base_url
)
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace="my_namespace"
)

# Печатаем результаты
for text in texts:
    embedding = cached_embedder.embed_query(text.page_content)
    print(embedding)

