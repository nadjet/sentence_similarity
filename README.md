# Sentence Similarity with Bert vs SBert


## Motivation

We can compute the similarity between two sentences by calculating the similarity between their embeddings. Now how do we obtain those sentence embeddings?

A popular approach is to perform the mean or max averaging of the sentence word embeddings. In the case of Bert, another pooling technique is to use the [CLS] or [SEP] token embedding. [Bert-as-a-service](https://github.com/hanxiao/bert-as-service) provides just the API to perform this kind of sentence encoding.

A more elaborate approach is proposed with [SBert](https://github.com/UKPLab/sentence-transformers), aka Sentence Bert. SBert is a siamese architecture in which pairs of sentence embeddings, obtained as above given a Bert or another transformer model, are weighted in conjunction according to a classification task such as sentence similarity. The resulting pretrained models provide meaningful dense vector representations for sentences according to the specific task they were fine-tuned on. This approach was shown to be faster and more performant than plain pooling of sentence word embeddings.

In this code I compute the similarity between a set of queries and a set of sentences, outputing the top k most similar sentences to each query. I do so with Bert-as-service on the one hand and SBert on the other hand, so as to compare the outputs and execution times.

Because of confidentiality issues, I do not provide the example texts, but I do include a brief summary of my results. I invite you to test the code with your own data.

## Getting started

The code was run with Python 3.7. You must install packages in `requirements.txt`. 

There are two main programs to compute similarity between each query and the set of sentences.

- `bertasservice_sim.py` using Bert-as-service
- `sembeds_sim.py` using SBert

For [Bert-as-a-service](https://github.com/hanxiao/bert-as-service) , you must install the server and client as specified on the page. You need to download the Bert model you are going to serve with. For my CPU I used the smaller model `uncased_L-12_H-768_A-1``. The bert service can be started as follows:

```
bert-serving-start -model_dir /path/to/uncased_L-12_H-768_A-12 -num_worker=4 -cpu
```

Both similarity programs take 3 arguments as input:

1. A text file containing the sentences, one per line.
2. A text file containing the queries for which we want to find the most similar sentences.
3. The output csv file that stores `<query,sentence,score>`.

We use Bert-as-service default which uses average mean pooling to encode sentences. Both programs use the cosine similarity metric and output the top 5 most similar sentences for each query.

## Results

The sentences were in the domain of survey questions for a client's project. These sentences are characterized with very similar sentence structure ("How satisfied are you with the quality of parks?", "How satisfied are you with the quality of health care?").

I had 662 sentences and 10 queries taken at random from those sentences. Bert-as-service took 22 seconds to compute the similarity whilst SBert took 7 seconds (MacOS 10.13).

For each <`<query,sentence,score>`, I evaluated whether:
1. At least one of the top 5 was relevant
2. The top first was relevant
3. Any of the top 5 were relevant

The results for Bert-as-a-service were 5/10, 3/10, 5/50.

For SBert the results were: 7/10, 6/10,14/50. So they were clearly superior to results with Bert-as-a-service.