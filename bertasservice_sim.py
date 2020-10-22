import numpy as np
from bert_serving.client import BertClient
import torch
import pandas as pd
import time
import plac
from utils.similarity import get_query_top_k
from utils.log import logger

@plac.annotations(
  sentences_file="Text file in which sentences are stored, one per line",
  queries_file="Text file in which sentences for which to find top 5 most similar sentences are stored, one per line",
  output_file="Output csv file containing query, sentence and score"
  )
def main(sentences_file, queries_file, output_file):
  start = time.time()

  bc = BertClient(check_length=False)


  logger.info("Loading sentences and queries...")
  with open(sentences_file,"r") as f:
    corpus = list(set([line.strip() for line in f.readlines()]))

  with open(queries_file,"r") as f:
    queries = [line.strip() for line in f.readlines()]


  logger.info("Encoding sentences...")
  doc_vecs = bc.encode(corpus)

  n = 10
  top_k = 5

  logger.info("Computing top {} similar sentences to each of {} queries...".format(top_k,len(queries)))


  data = []
  for query in queries:
      query_vec = bc.encode([query])[0]
      top_k_list = get_query_top_k(query, query_vec, corpus, doc_vecs, max_n = n, top_k = top_k)
      data.extend(top_k_list)
  df = pd.DataFrame(data)
  df.to_csv(output_file,index=False,sep="\t")
  end = time.time()

  e = int(end - start)
  logger.info('Time elapsed is: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

if __name__ == '__main__':
    plac.call(main)