import torch
import heapq

def eval(embeddings, labels, num_sample):
    """
    :param embeddings: list of tensors,all the embeddings
    :param labels: all the labels associate with the embeddings
    :param num_sample: how many neighbours are we drawing
    :return:
    """
    match_count=0
    for i,emb in enumerate(embeddings):
        temp = []
        for i2,emb2 in enumerate(embeddings):
            if i == i2:
                continue
            sim=torch.nn.functional.cosine_similarity(emb,emb2)
            heapq.heappush(temp,[sim,labels[i2]])
            if len(temp)>num_sample:
                heapq.heappop(temp)
        match_count += [l[1] for l in temp].count(labels[i])

    return float(match_count)/(len(embeddings)*num_sample)