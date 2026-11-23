#-------------------------------------------------------------------------
# AUTHOR: Jason Le
# FILENAME: vsm_search.py
# SPECIFICATION: Vector Space Model search engine. Ranks 100 documents for
#                each query by TF-IDF cosine similarity, computes Average
#                Precision (AP) per query from the relevance judgments, sorts
#                the queries by AP, and reports Mean Average Precision (MAP)
# FOR: CS 5180- Assignment #3
# TIME SPENT: 5 hours
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
docs = pd.read_csv("docs.csv")
queries = pd.read_csv("queries.csv")
judgments = pd.read_csv("relevance_judgments.csv")

doc_ids = docs["doc_id"].tolist()

# ---------------------------------------------------------
# 2. Build the TF-IDF matrix for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
vectorizer = TfidfVectorizer(stop_words="english")
doc_matrix = vectorizer.fit_transform(docs["text"])

# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
ap_values = {}

for _, query in queries.iterrows():
    query_id = query["query_id"]

    # represent the query with the same vocabulary and IDF values as the documents
    query_vector = vectorizer.transform([query["query_text"]])

    # score every document with cosine similarity
    scores = cosine_similarity(query_vector, doc_matrix).flatten()

    # rank all documents from highest to lowest score (ties keep document order)
    ranking = sorted(range(len(doc_ids)), key=lambda i: scores[i], reverse=True)
    ranked_doc_ids = [doc_ids[i] for i in ranking]

    # relevant documents for this query
    relevant = set(judgments[(judgments["query_id"] == query_id) &
                             (judgments["judgment"] == "R")]["doc_id"])

    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
    relevant_found = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            relevant_found += 1
            precision_sum += relevant_found / rank

    ap = precision_sum / len(relevant) if relevant else 0.0

    # store the AP value for this query (use any data structure you prefer)
    ap_values[query_id] = ap


# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
sorted_queries = sorted(ap_values.items(), key=lambda item: item[1], reverse=True)

mean_ap = sum(ap_values.values()) / len(ap_values)

# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")
query_text = dict(zip(queries["query_id"], queries["query_text"]))
for query_id, ap in sorted_queries:
    print(f"{query_id} ({query_text[query_id]}): AP = {ap:.4f}")
print("====================================================")
print(f"Mean Average Precision (MAP): {mean_ap:.4f}")