from hipo_rank import Similarities, Scores
import torch 
from numpy import ndarray
import numpy as np 

def lookup_raw_id(sec_id, sent_id, table):
    return table["-".join([str(sec_id), str(sent_id)])]

def _compute_similarities(embeds1: ndarray, embeds2: ndarray):
        embeds1 = torch.from_numpy(embeds1)
        embeds2 = torch.from_numpy(embeds2)
        similarities = torch.cosine_similarity(embeds1, embeds2).numpy()
        similarities = similarities / 2 + 0.5 # normalize to a range [0,1]
        similarities = np.clip(similarities, 0, 1)
        return similarities

class DynamicSentScorer:
    # adds sent_to_sect edges for each sentence
    def __init__(self,
                 forward_weight: float = 0.,
                 backward_weight: float = 1.,
                 section_weight: float = 1.,
                 sent_to_sent_weight: float = 0.1,
                 sent_to_sect_weight: float = 0.5,
                 sparse=False
                 ):
        # TODO: get rid of these god awful variable names
        self.forward_sent_to_sent_weight = forward_weight
        self.forward_sent_to_sect_weight = forward_weight * section_weight
        self.backward_sent_to_sent_weight = backward_weight
        self.backward_sent_to_sect_weight = backward_weight * section_weight

        self.sent_to_sent_weight = sent_to_sent_weight
        self.sent_to_sect_weight = sent_to_sect_weight
        self.sparse = sparse
        
    def get_raw_scores(self, similarities: Similarities, sparse=False) -> Scores:
        # build empty scores, indexed by scores[section_index][sentence_index]
        scores = []
        for sent_to_sent in similarities.sent_to_sent:
            if sent_to_sent.pair_indices:
                num_sents = max([x[1] for x in sent_to_sent.pair_indices]) + 1
            else:
                num_sents = 1
            scores.append([0 for _ in range(num_sents)])

        # add sent_to_sent scores
        for sect_index, sent_to_sent in enumerate(similarities.sent_to_sent):
            pids = sent_to_sent.pair_indices
            dirs = sent_to_sent.directions
            sims = sent_to_sent.similarities
            norm_factor = len(scores[sect_index]) - 1
            
            # 
            for ((i,j), dir, sim) in zip(pids, dirs, sims):
                if dir == "forward":
                    scores[sect_index][i] += self.forward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                    scores[sect_index][j] += self.backward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                elif dir == "backward":
                    scores[sect_index][j] += self.forward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                    scores[sect_index][i] += self.backward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                else:
                    scores[sect_index][j] += self.backward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                    scores[sect_index][i] += self.backward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight

        # add sent_to_sect scores
        for sect_index, sent_to_sect in enumerate(similarities.sent_to_sect):
            pids = sent_to_sect.pair_indices
            dirs = sent_to_sect.directions
            sims = sent_to_sect.similarities
            norm_factor = len(scores)
            for ((i,j), dir, sim) in zip(pids, dirs, sims):
                if dir == "forward":
                    scores[sect_index][i] += self.forward_sent_to_sect_weight * sim / norm_factor * self.sent_to_sect_weight
                elif dir == "backward" or dir == "undirected":
                    scores[sect_index][i] += self.backward_sent_to_sect_weight * sim / norm_factor * self.sent_to_sect_weight
        
        return scores 
    
    def rank_raw_scores(scores):
        ranked_scores = []
        sect_global_idx = 0
        for sect_idx, sect_scores in enumerate(scores):
            for sent_idx, sent_score in enumerate(sect_scores):
                ranked_scores.append(
                    (sent_score,
                     sect_idx,
                     sent_idx,
                     sect_global_idx + sent_idx
                     )
                )
            sect_global_idx += len(sect_scores)

        ranked_scores.sort(key=lambda x: x[0], reverse=True)
        return ranked_scores
            
    
    def get_updated_scores(self, raw_scores, similarities: Similarities, selected_list) -> Scores:
        # build empty scores, indexed by scores[section_index][sentence_index]
        scores = raw_scores.copy()
        
        
        for sect_index, sent_to_sect in enumerate(similarities.sent_to_sect):
            
            pids = sent_to_sect.pair_indices
            dirs = sent_to_sect.directions
            sims = sent_to_sect.similarities
            print(len(sims))
            if len(sims) > 500:
                continue
            norm_factor = len(scores[sect_index]) - 1
            print("Start Section", sect_index, len(pids))
            # for each sentence in current section.
            for ((i,j), dir, sim) in zip(pids, dirs, sims):
                # print(i)
                for item in selected_list:
                    selected_sec_idx, selected_local_idx = item.split("_")
                    selected_sec_idx = int(selected_sec_idx)
                    selected_local_idx = int(selected_local_idx)
                    
                # penalizing sec to sec.
                    for ((i_,j_), sim_) in zip(similarities.sect_to_sect.pair_indices, similarities.sect_to_sect.similarities):
                        if i_ == sect_index and j_ == selected_sec_idx:
                            scores[sect_index][i] -= self.backward_sent_to_sect_weight * sim_ / norm_factor * self.sent_to_sect_weight * 0.3
                            break
                    # rewarding the similarity to the previous sentence.
                    last_global_id = lookup_raw_id(selected_sec_idx, selected_local_idx, similarities.global_index_mapping)
                    cur_global_id = lookup_raw_id(sect_index, i, similarities.global_index_mapping)
                    
                    try:
                        idx_pair = similarities.all_sent_to_sent.pair_indices.index((last_global_id, cur_global_id))
                        scores[sect_index][i] += self.backward_sent_to_sect_weight * similarities.all_sent_to_sent.similarties[idx_pair] / norm_factor * self.sent_to_sent_weight * 0.3
                        break 
                    except:
                        pass 
               
        ranked_scores = []
        sect_global_idx = 0
        for sect_idx, sect_scores in enumerate(scores):
            for sent_idx, sent_score in enumerate(sect_scores):
                ranked_scores.append(
                    (sent_score,
                     sect_idx,
                     sent_idx,
                     sect_global_idx + sent_idx
                     )
                )
            sect_global_idx += len(sect_scores)

        ranked_scores.sort(key=lambda x: x[0], reverse=True)
        return ranked_scores

