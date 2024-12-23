from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import log
import string
import torch
from sentence_transformers import SentenceTransformer
import logging
import os


###Helper Functions####
# ---------#---------#---------#---------#---------#---------#---------
def normalize_string(input_string: str) -> str:
    translation_table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    string_without_punc = input_string.translate(translation_table)
    string_without_double_spaces = " ".join(string_without_punc.split())
    return string_without_double_spaces.lower()


def update_url_scores(old: dict[str, float], new: dict[str, float]):
    for url, score in new.items():
        old[url] = old.get(url, 0.0) + score
    return old


# ---------#---------#---------#---------#---------#---------#---------


##Main Search engine class
class SearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._index: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._documents: dict[str, str] = {}
        self.k1 = k1
        self.b = b
        self.logger = logging.getLogger(__name__)

        try:
            # This is the model we will be using for creating word embeddings for the query and docuement expansion!
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            self._vocab = set()  # For query embedding
            self._embedding_matrix = None  # used to store the embeddings for the vocabulary terms (words) that are present in the indexed documents.
            self._word_list = []  # Used in the _build_embeddings: populated with the sorted list of unique vocabulary terms from the indexed documents.

            ##If the file in load_embeddings does not exist then we execut the _bulding_embeddings function which start the word generation mechanism for doc and vocab embeddings.
            try:
                self.load_embeddings()
            except Exception as e:
                self.logger.warning(f"Could not load existing embeddings: {e}")
        ##I had issues because my computer would freeze up and crash, so I included an additional exception block!
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    """
    To my understanding:
    The posts() property returns a list of indexed document URLs.
    The number_of_documents() property gives the total count of indexed documents.
    The avdl property calculates the average length of indexed documents.
    These property attributes have already been created by Alex Molas.
    """

    @property
    def posts(self) -> list[str]:
        return list(self._documents.keys())

    @property
    def number_of_documents(self) -> int:
        return len(self._documents)

    @property
    def avdl(self) -> float:
        if not hasattr(self, "_avdl"):
            self._avdl = sum(len(d) for d in self._documents.values()) / len(self._documents)
        return self._avdl

    # IDF and bm25 standard implmentation left unmodified.
    def idf(self, kw: str) -> float:
        N = self.number_of_documents
        n_kw = len(self.get_urls(kw))
        return log((N - n_kw + 0.5) / (n_kw + 1) + 1)

    def bm25(self, kw: str) -> dict[str, float]:
        result = {}
        idf_score = self.idf(kw)
        avdl = self.avdl
        for url, freq in self.get_urls(kw).items():
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * len(self._documents[url]) / avdl)
            result[url] = idf_score * numerator / denominator
        return result

    # ---------#---------#---------#---------#---------#---------#---------
    # This is used to retrieve urls which is a nested dictionary that maps keywords
    # to a dictionary of URLs and their term frequencies. Used in IDF and BM25
    def get_urls(self, keyword: str) -> dict[str, int]:
        keyword = normalize_string(keyword)
        return self._index[keyword]

    # ---------#---------#---------#---------#---------#---------#---------
    # My Implementation
    """This code takes the document content provided by the index() and converts it into a tensor representation, 
    known as an embedding (also used in NlP), using the model.encode(content, convert_to_tensor=True) method. By 
    doing this, the code can measure using the cosine similarity. After creating the embedding for your input text, 
    it checks if there are enough documents in the _documents dictionary to compare against. If so, it grabs the 
    first 50 of those documents [for performance reasons my cpu can't handle any more than this:( ] , converts each 
    one into an embedding, and calculates the similarity between your input’s embedding and each stored document’s 
    embedding. Once the code finds the document with the highest similarity score, it checks if that score is above 
    0.5, now this threshold can be modified with larger datasets, I experimented using the test cases found in 
    test_engine.py and found this to be the highest performing weight. If it is, the text from the most similar 
    document gets appended to your input text, effectively “expanding” your original content. If something goes wrong 
    during the process, or if there aren’t any comparable documents, the code logs an error or warning and returns 
    your original text (worst case and good practice to do in real life applications)."""

    def document_expansion(self, content: str) -> str:

        try:

            doc_embedding = self.model.encode(content, convert_to_tensor=True)

            expanded_content = content

            if len(self._documents) > 1:
                try:
                    # This is where the first 50 are assigned doc_texts!
                    doc_texts = list(self._documents.values())[:100]
                    doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)

                    # We compute their similarities , this is our ranking algorithm basically
                    similarities = torch.nn.functional.cosine_similarity(
                        doc_embedding.unsqueeze(0),
                        # This operation adds an extra dimension to the document embedding tensor,
                        # transforming it from a 1D vector
                        # to a 2D tensor with a batch dimension of 1. This is necessary because the torch.nn.CS expects a "batch" similarity.
                        # I learned this the hard way took me an hour to find this out!
                        doc_embeddings
                    )

                    top_idx = similarities.argmax().item()
                    if similarities[top_idx] > 0.5:  # Only use if similarity is high enough, refer to the text above!
                        expanded_content = f"{content} {doc_texts[top_idx]}"  # if it then we add it to expanded_content"
                except Exception as e:
                    self.logger.warning(f"Error in similarity calculation: {e}")

            return expanded_content

        except Exception as e:
            self.logger.error(f"Error in document expansion: {e}")
            return content  # Return original content if expansion fails

    # index() was created by Alex Molas and it is responsible for adding a new document to the search index.

    def index(self, url: str, content: str) -> None:
        expanded_content = self.document_expansion(content)
        self._documents[url] = expanded_content
        words = normalize_string(expanded_content).split(" ")
        for word in words:
            self._index[word][url] += 1
            self._vocab.add(word)
        if hasattr(self, "_avdl"):
            del self._avdl

    """The bulk_expand_documents method was created by me for expanding the content of multiple documents in parallel 
    using document expansion. This processes the urls in chunks and uses multi-threading."""

    def bulk_expand_documents(self, documents):

        expanded_documents = []

        def expand_single_document(doc):
            try:
                return {
                    "url": doc[0],
                    "expanded_content": self.document_expansion(doc[1])
                }
            except Exception as e:
                self.logger.error(f"Error expanding document {doc[0]}: {e}")
                return {
                    "url": doc[0],
                    "expanded_content": doc[1]
                    # This is the fallback from if there is no comparable documents (from document_expansion).
                }

        chunk_size = 8000
        total_docs = len(documents)

        for i in range(0, total_docs, chunk_size):
            chunk = documents[i:i + chunk_size]
            try:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(expand_single_document, doc) for doc in chunk]
                    for future in futures:
                        try:
                            result = future.result(timeout=600)
                            if result:
                                expanded_documents.append(result)
                        except Exception as e:
                            self.logger.error(f"Error processing document in chunk: {e}")
                            continue

                self.logger.info(f"Processed {min(i + chunk_size, total_docs)} / {total_docs} documents")

            except Exception as e:
                self.logger.error(f"Error processing chunk: {e}")
                continue

        return expanded_documents

    """The bulk_index method is responsible the collection of docs by checking if there are any previously expanded 
    documents stored in the expanded_path file, I implmented this because previously when you run the app it would 
    build embeddings all over again but now the embeddings are saved and loaded. If there exist the 
    expanded_document.pt file, it loads those expanded documents and skips the expensive expansion process. If no 
    expanded documents are found, the method continues to process the documents in chunks. It does this by calling 
    the bulk_expand_documents method on each chunk which uses the expanded_document formula, which expands the 
    content of the documents in parallel to improve performance. After the expansion, the method iterates through the 
    expanded documents and calls the index method on each one. This adds the document to the search index, 
    building the necessary data structures like the inverted index and the vocabulary, which were two main concepts 
    used in IR lecture! :)"""

    def bulk_index(self, documents: list[tuple[str, str]], chunk_size: int = 1000,
                   expanded_path: str = "expanded_documents.pt"):

        try:
            if os.path.exists(expanded_path):
                self.logger.info(f"Found existing expanded documents at {expanded_path}. Loading...")
                self.load_expanded_documents(expanded_path)
            else:
                self.logger.info(f"No expanded documents found at {expanded_path}. Expanding and indexing...")
                expanded_documents = []
                total_docs = len(documents)

                for i in range(0, total_docs, chunk_size):
                    chunk = documents[i:i + chunk_size]
                    try:
                        expanded_chunk = self.bulk_expand_documents(chunk)
                        expanded_documents.extend(expanded_chunk)

                        for doc in expanded_chunk:
                            try:
                                self.index(doc["url"], doc["expanded_content"])
                            except Exception as e:
                                self.logger.error(f"Error indexing document {doc['url']}: {e}")
                                continue

                        self.logger.info(f"Indexed {min(i + chunk_size, total_docs)} / {total_docs} documents")

                        if i % (chunk_size * 5) == 0:
                            self.save_expanded_documents(expanded_documents, expanded_path)

                    except Exception as e:
                        self.logger.error(f"Error processing chunk starting at index {i}: {e}")
                        continue

                self.save_expanded_documents(expanded_documents, expanded_path)
            # This provides progress tracking, helpful to know what is going on in the process on your CLI
            if self._embedding_matrix is None:
                self.logger.info("Building embeddings...")
                self._build_embeddings()
                self.logger.info("Embeddings built successfully")

        except Exception as e:
            self.logger.error(f"Critical error during bulk indexing: {e}")
            raise

    """It first ensures the directory project directory exist, I implemented this function outside of the project , 
    then saves the expanded_documents list to the specified file path using PyTorch's torch.save() function. If any 
    errors occur during the saving process, it logs the error and raises the exception. Standard practice"""

    def save_expanded_documents(self, expanded_documents, file_path="expanded_documents.pt"):
        try:

            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)

            # expanded_documents.pt contains actual text (the updated, expanded text for each document).
            torch.save(expanded_documents, file_path)
            self.logger.info(f"Expanded documents saved to {file_path}")

        except Exception as e:
            self.logger.error(f"Error saving expanded documents to {file_path}: {e}")
            raise

    """The load_expanded_documents method loads expanded documents from a torch file. It checks if the file exists, 
    and if so, it loads the expanded documents and stores them in the self._documents dictionary. If the file is not 
    found, it logs a warning."""

    def load_expanded_documents(self, file_path="expanded_documents.pt"):
        try:
            if os.path.exists(file_path):
                expanded_documents = torch.load(file_path)
                for doc in expanded_documents:
                    self._documents[doc["url"]] = doc["expanded_content"]
                self.logger.info(f"Loaded expanded documents from {file_path}")
            else:
                self.logger.warning(f"No expanded documents found at {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading expanded documents from {file_path}: {e}")
            raise

    """The _build_embeddings method generates and saves document and vocabulary embeddings all in one sweep. It encodes the document 
    contents and vocabulary terms using the SentenceTransformer model made by huggingface, storing the resulting 
    embeddings in the self._doc_embeddings and self._embedding_matrix attributes. These embeddings are then saved to 
    the project folder"""

    def _build_embeddings(self):

        def encode_batch(texts):
            try:
                return self.model.encode(texts, convert_to_tensor=True, batch_size=96)
            except Exception as e:
                self.logger.error(f"Batch encoding error: {e}")
                return None

        # Process all texts in one go
        all_texts = list(set(self._documents.values()))
        self.logger.info(f"Processing {len(all_texts)} unique documents...")

        # Process documents in larger batches, process took to long
        batch_size = 4000
        doc_embeddings = []


        max_workers = min(6, (os.cpu_count() or 1))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i:i + batch_size]
                futures.append(executor.submit(encode_batch, batch))

            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=1200)  # Increased timeout
                    if result is not None:
                        doc_embeddings.append(result)
                    self.logger.info(f"Processed document batch {i + 1}/{len(futures)}")
                except Exception as e:
                    self.logger.error(f"Error processing document batch: {e}")

        if doc_embeddings:
            self._doc_embeddings = torch.cat(doc_embeddings, dim=0)
            torch.save(self._doc_embeddings, 'document_embeddings.pt')

        # Build vocab embeddings only if needed for query expansion
        word_counts = {word: len(self.get_urls(word)) for word in self._vocab}
        # Increased threshold to reduce vocabulary size
        frequent_words = [word for word, count in word_counts.items()
                          if count > len(self._documents) * 0.02]  # Increased from 0.01
        self._word_list = sorted(frequent_words)

        if self._word_list:
            self._embedding_matrix = encode_batch(self._word_list)
            if self._embedding_matrix is not None:
                self.save_embeddings()

    """The load_embeddings method loads the previously saved document and vocabulary embeddings from project dir. It checks 
    if the expected files exist, and if so, it loads the embeddings and assigns them to the relevant attributes."""

    def load_embeddings(self, doc_path: str = 'document_embeddings.pt', vocab_path: str = 'embeddings.pt'):

        try:

            if os.path.exists(doc_path):
                """_doc_embeddings is defined outside of the constructor to allow the embeddings to be loaded or 
                generated as needed, rather than requiring them to be present when the SearchEngine class is 
                instantiated. I refer to this as the modular architecture."""

                self._doc_embeddings = torch.load(doc_path)
                self.logger.info(f"Document embeddings loaded from {doc_path}")
            else:
                self.logger.warning(f"No document embeddings found at {doc_path}")

            if os.path.exists(vocab_path):
                data = torch.load(vocab_path)
                if isinstance(data, dict) and 'embedding_matrix' in data and 'word_list' in data:
                    self._embedding_matrix = data['embedding_matrix']
                    self._word_list = data['word_list']
                    self._vocab = set(self._word_list)
                    self.logger.info(f"Vocabulary embeddings loaded from {vocab_path}")
                else:  # Progress checks and expcetion warnings put in place. I like to know what is going for many cases like fallback or variable mishandling
                    self.logger.error(f"Invalid data format in {vocab_path}")
            else:
                self.logger.warning(f"No vocabulary embeddings found at {vocab_path}")

        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            raise

    """The save_embeddings method saves the current document and vocabulary embeddings to disk. It first checks if 
    the required embedding data is available, and then writes the embeddings to the specified file paths."""

    def save_embeddings(self, doc_path: str = 'document_embeddings.pt', vocab_path: str = 'embeddings.pt'):

        try:
            # Save document embedding it
            if hasattr(self, '_doc_embeddings'):
                torch.save(self._doc_embeddings, doc_path)
                self.logger.info(f"Document embeddings saved to {doc_path}")

                # Save word embeddings
            if self._embedding_matrix is not None and self._word_list:
                torch.save({
                    'embedding_matrix': self._embedding_matrix,
                    'word_list': self._word_list
                }, vocab_path)
                self.logger.info(f"Vocabulary embeddings saved to {vocab_path}")
            else:
                self.logger.warning("No embeddings to save")

        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
            raise

    """The calculate_embedding_similarity method computes the cosine similarity between a given query and the 
    document embeddings. This allows the search engine to identify semantically relevant documents for the query, 
    beyond just lexical matching."""

    def calculate_embedding_similarity(self, query: str) -> dict[str, float]:

        if not hasattr(self, '_doc_embeddings'):
            self.logger.error("Document embeddings not built yet")
            return {}

        try:

            query_embedding = self.model.encode(query, convert_to_tensor=True)

            # We use cosine_similarity
            similarity_scores = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                self._doc_embeddings
            )

            # We assign the similarity scores to the document. '_documents' is a nested dictionary. At this point it
            # contains the mapping between document URLs and their corresponding expanded content.
            result = {}
            for i, url in enumerate(self._documents.keys()):
                result[url] = float(similarity_scores[i])

            return result

        except Exception as e:
            self.logger.error(f"Error calculating embedding similarity: {e}")
            return {}

    """The expand_query method expands the original query by finding semantically similar terms using the vocabulary 
    embeddings. This helps the search engine capture a broader range of relevant content. This is the main function 
    for the query expansion feature."""

    def expand_query(self, keywords: list[str]) -> list[str]:

        expanded_keywords = []
        expanded_keywords.extend(keywords)

        for word in keywords:
            query_emb = self.model.encode(word, convert_to_tensor=True)

            if self._embedding_matrix is not None and self._word_list:
                """The query expansion is done at runtime, during the search process. The expand_query method 
                calculates the cosine similarity between the query terms and the vocabulary embeddings stored in 
                self._embedding_matrix..."""
                similarities = torch.nn.functional.cosine_similarity(
                    query_emb.unsqueeze(0),
                    self._embedding_matrix  # in here!
                )

                # This is a hyperparameter that COULD be tuned.
                top_k = 6
                top_indices = torch.topk(similarities, k=min(top_k + 1, len(self._word_list))).indices

                # This is later passed to the search() method see below
                for idx in top_indices:
                    term = self._word_list[idx]
                    if term not in expanded_keywords:
                        expanded_keywords.append(term)

        return expanded_keywords

    """The search method is the last method for embedding creation and the first method triggered by the user query. 
    It starts by normalizing the input query and splitting it into keywords, which are then expanded to include 
    related terms. Both the original and expanded keywords are logged for reference. Using the BM25 algorithm, 
    the method calculates relevance scores for URLs based on the original keywords, giving them a higher weight of 
    1.2. For the expanded keywords, it computes scores with a reduced weight of 0.8. Additionally, it incorporates 
    embedding similarity scoring to assess the semantic relevance of the URLs. Finally, the method combines the BM25 
    and embedding scores with adjusted weights—60% for BM25 and 40% for embeddings—sorts the results in descending 
    order, and returns the most relevant URLs! This took awhile to implement around 3 days of figuring it out :)"""

    def search(self, query: str) -> dict[str, float]:

        original_keywords = normalize_string(query).split(" ")
        expanded_keywords = self.expand_query(original_keywords)

        self.logger.info(f"Original keywords: {original_keywords}")
        self.logger.info(f"Expanded keywords: {expanded_keywords}")


        bm25_scores = {}
        # Higher weight for original terms,
        # because we give more importance to the user query, rather than pseudo generated expansions.
        for kw in original_keywords:
            kw_scores = self.bm25(kw)
            bm25_scores = update_url_scores(bm25_scores, {url: score * 1.2 for url, score in kw_scores.items()})

        """
                In case you are wondering what this looks like:
                bm25_scores = {
                    'https://example.com/rss/feed/document1': 0.8,
                    'https://example.com/blog/feed/document2': 1.2,
                    'https://example.com/blogs/feed/document3': 0.6,
                    'https://example.com/rss/feed/document4': 1.0,
                    'https://example.com/type/?rss/document5': 0.9
                }
        """


        for kw in expanded_keywords:
            if kw not in original_keywords:
                kw_scores = self.bm25(kw)
                bm25_scores = update_url_scores(bm25_scores, {url: score * 0.8 for url, score in kw_scores.items()})



        # Embedding similarity scoring, this uses document_embeddings.pt!
        embedding_scores = self.calculate_embedding_similarity(query)

        # Combine scores with adjusted weights
        combined_scores = {}
        for url in set(bm25_scores.keys()).union(embedding_scores.keys()):
            bm25 = bm25_scores.get(url, 0)
            embedding = embedding_scores.get(url, 0)
            combined_scores[url] = 0.6 * bm25 + 0.4 * embedding  # Adjusted weights that I've found to be the best.

        return dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))

