from threading import local
from hipo_rank import Scores, Document, Similarities, Summary
from hipo_rank.scorers.dynamic import DynamicScorer


class DefaultSummarizer:
    def __init__(self, num_words: int = 200, stay_under_num_words: bool = False):
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words

    def get_summary(self, doc: Document, sorted_scores: Scores) -> Summary:
        num_words = 0
        summary = []
        i = 0
        while True:
            sect_idx = sorted_scores[i][1]
            local_idx = sorted_scores[i][2]
            sentence = doc.sections[sect_idx].sentences[local_idx]
            num_words += len(sentence.split())
            if self.stay_under_num_words and num_words > self.num_words:
                break
            summary.append((sentence, *sorted_scores[i]))
            i += 1
            if num_words >= self.num_words:
                break
            if i >= len(sorted_scores):
                break
        return summary

class DynamicSummarizer:
    def __init__(self, num_words: int = 220, stay_under_num_words: bool = False, start_ratio=0.3):
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words
        self.start_ratio = start_ratio

    def get_summary(self, doc: Document, sorted_scores: Scores, sim, dscorer) -> Summary:
        num_words = 0
        summary = []
        i = 0
        print("Start selecting summary ... ")
        
        selected = []
        raw_scores = dscorer.get_raw_scores(sim)
        while True:
            if num_words < self.start_ratio * self.num_words:
                
                sect_idx = sorted_scores[i][1]
                local_idx = sorted_scores[i][2]
                
                appeared = "_".join([str(sect_idx), str(local_idx)])
                if appeared not in selected:
                    selected.append(appeared)
                
                sentence = doc.sections[sect_idx].sentences[local_idx]
                num_words += len(sentence.split())
                print(num_words)
                if self.stay_under_num_words and num_words > self.num_words:
                    break
                summary.append((sentence, *sorted_scores[i]))
                i += 1
                if num_words >= self.num_words:
                    break
                if i >= len(sorted_scores):
                    break
            else:
                # dynamic selection.
                i = 0 
                ranked_scores = dscorer.get_updated_scores(raw_scores, sim, sect_idx, local_idx)
                print(ranked_scores)
                for i in range(len(ranked_scores)):
                    score, sect_idx, local_idx, _ = ranked_scores[i]
                    appeared = "_".join([str(sect_idx), str(local_idx)])
                    if i >= len(sorted_scores):
                        break
                    if appeared not in selected:
                        selected.append(appeared)
                        print("Dynamic selected ..", sect_idx, local_idx)
                        print(num_words)
                        sentence = doc.sections[sect_idx].sentences[local_idx]
                        num_words += len(sentence.split())
                        break
                    else:
                        continue
                if self.stay_under_num_words and num_words > self.num_words:
                    break
                if i >= len(sorted_scores)-1:
                    break
                summary.append((sentence, score))
                
                if num_words >= self.num_words:
                    break
                   
                
        return summary
    
    
class DynamicSentSummarizer:
    def __init__(self, num_words: int = 220, stay_under_num_words: bool = False, start_ratio=0.3):
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words
        self.start_ratio = start_ratio

    def get_summary(self, doc: Document, sorted_scores: Scores, sim, dscorer) -> Summary:
        num_words = 0
        summary = []
        i = 0
        print("Start selecting summary ... ")
        
        selected = []
        
        selected_pairs = []
        raw_scores = dscorer.get_raw_scores(sim)
        while True:
            if num_words < self.start_ratio * self.num_words:
                
                sect_idx = sorted_scores[i][1]
                local_idx = sorted_scores[i][2]
                
                appeared = "_".join([str(sect_idx), str(local_idx)])
                if appeared not in selected:
                    selected.append(appeared)
                    selected_pairs.append(appeared)
                sentence = doc.sections[sect_idx].sentences[local_idx]
                num_words += len(sentence.split())
                print(num_words)
                if self.stay_under_num_words and num_words > self.num_words:
                    break
                summary.append((sentence, *sorted_scores[i]))
                i += 1
                if num_words >= self.num_words:
                    break
                if i >= len(sorted_scores):
                    break
            else:
                # dynamic selection.
                i = 0 
                ranked_scores = dscorer.get_updated_scores(raw_scores, sim, selected_pairs)
                print(ranked_scores)
                for i in range(len(ranked_scores)):
                    score, sect_idx, local_idx, _ = ranked_scores[i]
                    appeared = "_".join([str(sect_idx), str(local_idx)])
                    if i >= len(sorted_scores):
                        break
                    if appeared not in selected:
                        selected.append(appeared)
                        print("Dynamic selected ..", sect_idx, local_idx)
                        print(num_words)
                        sentence = doc.sections[sect_idx].sentences[local_idx]
                        num_words += len(sentence.split())
                        break
                    else:
                        continue
                if self.stay_under_num_words and num_words > self.num_words:
                    break
                if i >= len(sorted_scores)-1:
                    break
                summary.append((sentence, score))
                
                if num_words >= self.num_words:
                    break
                   
                
        return summary


