## 1. Research Question

Can fine-grained BERT-based query-document interaction be retained
without running BERT jointly over every query-document pair at
search time?

ColBERT addresses this using contextualized late interaction.

## 2. High-Level Architecture

The retrieval pipeline is:

Query
  ↓
BERT Query Encoder
  ↓
Query token embeddings
  ↓
Linear projection
  ↓
Normalized query representations
              \
               → Late Interaction / MaxSim → Relevance Score
              /
Document
  ↓
BERT Document Encoder
  ↓
Document token embeddings
  ↓
Linear projection
  ↓
Normalized document representations

The document representations can be precomputed and indexed.

## 3. Main Difference From Other Retrieval Architectures

Cross-Encoder:
    Query + Document → BERT → Score

Bi-Encoder:
    Query → one vector
    Document → one vector
    similarity(query_vector, document_vector)

ColBERT:
    Query → multiple contextualized token vectors
    Document → multiple contextualized token vectors
    token-level late interaction → Score

ColBERT therefore attempts to combine:

1. Fine-grained token-level matching
2. Precomputed document representations
3. Efficient retrieval