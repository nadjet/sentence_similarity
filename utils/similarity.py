from sentence_transformers import util
import torch
import pandas as pandas

def similarity(query_embeddings,docs_embeddings,max_n=10, top_k=5):
	cos_scores = util.pytorch_cos_sim(query_embeddings, docs_embeddings)[0]
	cos_scores = cos_scores.cpu()
	top_results = torch.topk(cos_scores, k=max_n)
	return zip(top_results[0], top_results[1])

def get_query_top_k(query, query_embeddings, docs, docs_embeddings, max_n=10, top_k=5, min_p=0.7, exact_match=True):
	count=0
	top_k_list = []
	for score, idx in similarity(query_embeddings, docs_embeddings, max_n=max_n, top_k=top_k):
		score = score.item()
		if count<top_k and ((score>min_p and exact_match) or (score<=0.99 and score>min_p)): # we skip exact match if so required, because of floating point precision we set exact match to 0.99
			count=count+1
			top_k_list.append({"query":query,"sentence":docs[idx],"score":score})
	return top_k_list