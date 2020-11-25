from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import time
import pandas as pd
from utils.log import logger
from utils.similarity import get_query_top_k
import plac



@plac.annotations(
  sentences_file="Text file in which sentences are stored, one per line",
  queries_file="Text file in which sentences for which to find top 5 most similar sentences are stored, one per line",
  output_file="Output csv file containing query, sentence and score"
  )
def main(sentences_file, queries_file, output_file, threshold=0.8):

  start = time.time()

  embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

  logger.info("Loading sentences and queries...")

  with open(sentences_file,"r") as f:
    corpus = [line.strip() for line in f.readlines()]

  with open(queries_file,"r") as f:
    queries = [line.strip() for line in f.readlines()]

  logger.info("Encoding sentences...")
  corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

  # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
  n = 10
  top_k = 5

  logger.info("Computing top {} similar sentences to each of {} queries...".format(top_k,len(queries)))

  data = []
  for query in queries:
      query_embeddings = embedder.encode(query, convert_to_tensor=True)
      top_k_list = get_query_top_k(query,query_embeddings, corpus, corpus_embeddings, max_n = n, top_k = top_k, min_p=threshold)
      data.extend(top_k_list)
  df = pd.DataFrame(data)
  df.to_csv(output_file,index=False,sep="\t")
  end = time.time()

  e = int(end - start)
  logger.info('Time elapsed is: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

if __name__ == '__main__':
    plac.call(main)