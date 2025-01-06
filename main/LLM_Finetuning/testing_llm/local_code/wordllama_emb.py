from wordllama import WordLlama
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load the default WordLlama model
        wl = WordLlama.load()
        logger.info("WordLlama model loaded successfully.")

        # Calculate similarity between two sentences
        similarity_score = wl.similarity("i went to the car", "i went to the pawn shop")
        logger.info(f"Similarity score: {similarity_score}")

        # Rank documents based on their similarity to a query
        query = "i went to the car"
        candidates = ["i went to the park", "i went to the shop", "i went to the truck", "i went to the vehicle"]
        ranked_docs = wl.rank(query, candidates)
        logger.info(f"Ranked documents: {ranked_docs}")

        # Extract just the strings from ranked_docs for clustering
        doc_strings = [doc for doc, _ in ranked_docs]
        
        # Perform clustering
        try:
            clusters = wl.cluster(doc_strings, k=2, max_iterations=100, tolerance=1e-4)
            logger.info(f"Clustering result: {clusters}")
        except TypeError as e:
            logger.error(f"TypeError in clustering: {e}")
            logger.info(f"Input to cluster method: {doc_strings}")

        # Additional inference methods
        deduped = wl.deduplicate(candidates, threshold=0.8)
        logger.info(f"Deduplicated candidates: {deduped}")

        filtered = wl.filter(query, candidates, threshold=0.3)
        logger.info(f"Filtered candidates: {filtered}")

        topk = wl.topk(query, candidates, k=3)
        logger.info(f"Top K candidates: {topk}")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Please ensure WordLlama is installed correctly and there are no naming conflicts.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")







if __name__ == "__main__":
    # main()  
    from wordllama import WordLlama

    # Load the default WordLlama model
    wl = WordLlama.load()

    # Calculate similarity between two sentences
    similarity_score = wl.similarity("i went to the car", "i went to the pawn shop")
    # print(similarity_score)  # Output: 0.06641249096796882

    # Rank documents based on their similarity to a query
    query = "Determine the airway bill number in the Covering Schedule document named pdf11_8.png."
    candidates = [
    'CS: contains the covering schedule document information',
    'AWB: contains the airway bill document information',
    'BOE: contains the bill of exchange document information',
    'CI: contains the commercial invoice document information',
    'COO: contains the certificate of origin document information',
    'IC: contains the insurance certificate document information',
    'PL: contains the packing list document information',
    'classification_table: contains the document class information'
    ]
    
    # candidates = ["i went to the park", "i went to the shop", "i went to the truck", "i went to the vehicle"]
    
    ranked_docs = wl.rank(query, candidates)
    print('>>>>>>>>>>>>>.')
    print('>>>>>>>>>>>>>.')
    print(ranked_docs)
    print('>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>')
    wl.deduplicate(candidates, threshold=0.8) # fuzzy deduplication
    wl.cluster(candidates, k=5, max_iterations=100, tolerance=1e-4) # labels using kmeans/kmeans++ init
    deduped = wl.deduplicate(candidates, threshold=0.8)
    filtered = wl.filter(query, candidates, threshold=0.3)
    print(filtered)
    topk = wl.topk(query, candidates, k=3)
    print(topk)
    