import torch
import numpy as np

from hipo_rank import Embeddings, SentenceEmbeddings, SectionEmbedding, \
    PairIndices, SentenceSimilarities, SectionSimilarities, AllSimilarities, Similarities
from typing import List, Tuple
from numpy import ndarray


class BonusCosSimilarity:
    def __init__(self, threshold = 0):
        self.threshold = threshold

    def _compute_similarities(self, embeds1: ndarray, embeds2: ndarray) -> ndarray:
        embeds1 = torch.from_numpy(embeds1)
        embeds2 = torch.from_numpy(embeds2)
        similarities = torch.cosine_similarity(embeds1, embeds2).numpy()
        similarities = similarities / 2 + 0.5 # normalize to a range [0,1]
        similarities = np.clip(similarities, self.threshold, 1)
        return similarities

    def _get_pairwise_similarities(self, embeds: ndarray) -> Tuple[ndarray, PairIndices]:
        pair_indices = self._get_pair_indices(len(embeds))
        pair_indices_i = [x[0] for x in pair_indices]
        pair_indices_j = [x[1] for x in pair_indices]
        similarities = self._compute_similarities(embeds[pair_indices_i], embeds[pair_indices_j])
        return similarities, pair_indices

    def _get_pair_indices(self, num_nodes: int) -> PairIndices:
        pair_indices = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                pair_indices += [(i, j)]
        return pair_indices

    def get_similarities(self, embeds: Embeddings):
        np.random.seed(0)
        # sent2sent embedding
        sent_to_sent = []
        global_index_mapping = {}
        global_sec_idx = 0
        global_sent_idx = 0
        for sent_embeds in embeds.sentence:
            for i in range(len(sent_embeds.embeddings)):
                global_index_mapping["-".join([str(global_sec_idx), str(i)])] = global_sent_idx
                global_sent_idx += 1
            global_sec_idx += 1
            id = sent_embeds.id
            e = sent_embeds.embeddings
            
            if global_sec_idx == 1:
                all_sent_embeds = e
            else:
                all_sent_embeds = np.concatenate([all_sent_embeds, e], axis=0)
            similarities, pair_indices = self._get_pairwise_similarities(e)
            directions = ["undirected" for _ in pair_indices]
            sent_to_sent += [SentenceSimilarities(id, similarities, pair_indices, directions)]

        all_sents_similarities, all_pair_indices = self._get_pairwise_similarities(all_sent_embeds)
        all_directions = ["undirected" for _ in all_pair_indices]
        all_sent_to_sent = SentenceSimilarities(id, all_sents_similarities, all_pair_indices, all_directions)
        
        sent_to_sect = []
        sect_embeds = np.stack([s.embedding for s in embeds.section])
        num_sect = len(sect_embeds)
        
        
        for sent_embeds in embeds.sentence:
            # TODO: factor out pair indices for one and two matrices
            pair_indices = []
            num_sent = len(sent_embeds.embeddings)
            for i in range(num_sent):
                for j in range(num_sect):
                    pair_indices += [(i,j)]
            pair_indices_i = [x[0] for x in pair_indices]
            pair_indices_j = [x[1] for x in pair_indices]
            embeds1 = sent_embeds.embeddings[pair_indices_i]
            embeds2 = sect_embeds[pair_indices_j]
            similarities = self._compute_similarities(embeds1, embeds2)
            id = sent_embeds.id
            directions = ["undirected" for _ in pair_indices]
            sent_to_sect += [SentenceSimilarities(id, similarities, pair_indices, directions)]

        similarities, pair_indices = self._get_pairwise_similarities(sect_embeds)
        directions = ["undirected" for _ in pair_indices]
        sect_to_sect = SectionSimilarities(similarities, pair_indices, directions) 
         
        # print(global_index_mapping)
        return AllSimilarities(sent_to_sent, sect_to_sect, sent_to_sect, all_sent_to_sent, global_index_mapping)











