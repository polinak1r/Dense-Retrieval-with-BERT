# Dense Retrieval with BERT
This notebook implements a dense retrieval pipeline for semantic search using a pretrained RoBERTa model.  

1. **Data Preparation**  
   Load data. For quick experiments apply sampling.

2. **Model Initialization and Embedding Computation**  
   Load and explore RoBERTa from HuggingFace Transformers. Compute contextual embeddings for queries and documents via masked mean pooling over token-level outputs, followed by batch standardization and L2 normalization.

3. **Similarity Scoring and Retrieval Evaluation**  
   Compute pairwise similarities between query and document embeddings using dot product. Rank documents per query and evaluate retrieval performance using the PFound metric, which reflects ranked relevance under user browsing behavior.

4. **Inference and Submission**  
   Run the trained pipeline on test queries, generate predicted rankings, and export results in kaggle submission format.

You can find code at [notebook](https://github.com/polinak1r/Dense-Retrieval-with-BERT/blob/main/Dense_Retrieval_with_BERT.ipynb)
