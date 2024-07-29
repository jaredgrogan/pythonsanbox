import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize

class RAGModule:
    def __init__(self, dim, n_trees=10, metric='angular'):
        self.dim = dim
        self.n_trees = n_trees
        self.metric = metric
        self.index_faiss = faiss.IndexFlatL2(dim)  # Using L2 distance
        self.index_annoy = AnnoyIndex(dim, metric)
        self.tfidf_vectorizer = TfidfVectorizer()
        
    def fit(self, documents):
        # Fit TF-IDF vectorizer
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.tfidf_matrix = normalize(self.tfidf_matrix)  # Normalize

        # Convert TF-IDF matrix to dense format and numpy array
        self.tfidf_matrix_dense = self.tfidf_matrix.toarray()
        
        # Indexing with FAISS
        self.index_faiss.add(self.tfidf_matrix_dense.astype(np.float32))

        # Indexing with Annoy
        self.index_annoy.build(self.n_trees)
        for i, vec in enumerate(self.tfidf_matrix_dense):
            self.index_annoy.add_item(i, vec)
        
    def add_documents(self, documents):
        # Re-fit and re-index
        self.fit(documents)
        
    def query(self, query_text, top_k=10):
        # Process query
        query_vec = self.tfidf_vectorizer.transform([query_text])
        query_vec = normalize(query_vec).toarray()
        
        # Search with FAISS
        faiss_distances, faiss_indices = self.index_faiss.search(query_vec.astype(np.float32), top_k)
        
        # Search with Annoy
        annoy_indices = self.index_annoy.get_nns_by_vector(query_vec[0], top_k)
        
        # Combine results
        combined_indices = set(faiss_indices[0]).union(set(annoy_indices))
        
        # Re-ranking
        combined_indices = list(combined_indices)
        if len(combined_indices) == 0:
            return [], []

        # Compute similarity
        combined_vectors = self.tfidf_matrix_dense[combined_indices]
        similarities = cosine_similarity(query_vec, combined_vectors)
        
        # Sort by similarity
        sorted_indices = np.argsort(-similarities[0])
        top_indices = [combined_indices[i] for i in sorted_indices[:top_k]]
        
        return top_indices, similarities[0][sorted_indices[:top_k]]

# Usage example
documents = [
    "Machine learning is great for predictive analytics.",
    "Natural language processing can understand and generate human language.",
    "Vector databases are essential for efficient retrieval in AI systems.",
    "Hybrid search combines different search techniques for improved results."
]

rag = RAGModule(dim=100)  # Adjust dim to your embedding size
rag.fit(documents)

query = "How does machine learning aid in predictive analytics?"
top_indices, similarities = rag.query(query)

print("Top document indices:", top_indices)
print("Similarities:", similarities)
