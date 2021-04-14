from __future__ import division
import argparse
import pickle
import re
import pandas as pd

# useful stuff
import numpy as np
import string

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy.special import expit
from sklearn.preprocessing import normalize

__authors__ = ['Mathieu_Tardy', 'Jeremie_Feron', 'Benjamin Bonvalet', 'Tom Huix']
__emails__ = ['mathieu.tardy@student-cs.fr', 'jeremie.feron@student-cs.fr',
              'benjamin.bonvalet@supelec.fr', 'tom.huix@supelec.fr']

stop_words = set(stopwords.words('english'))


def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentence = l.lower().split()
            cleaned_sentence = [word.strip(string.punctuation).lower() for word in sentence if
                                word not in stop_words and not re.findall(r"\d", word)]
            cleaned_sentenced_without_empty = [word for word in cleaned_sentence if
                                               word != '' and len(word) != 1]
            sentences.append(cleaned_sentenced_without_empty)
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5, n_gram=3,
                 mode_OOV=False, alpha=1):
        self.w2id = {}  # set of word to ID mapping
        self.train_set = list()
        self.nEmbed = nEmbed  # size of matrix embedding
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.trainWords = 0
        self.accLoss = 0.
        self.unknown_vector = np.random.normal(0, 1, (self.nEmbed,))

        self.sentences = sentences
        self.loss = []  # Record losses in training
        self.l_r = 0.0002
        self.W = []  # Weight matrix
        self.C = []  # Context matrix
        self.alpha = alpha

        self.mode_OOV = mode_OOV

        if self.mode_OOV:
            # OOV
            self.n_gram = n_gram
            self.words_and_their_ngrams = []
            full_vocab = []  # words + subwords
            self.n_gram_w2id = {}

    def initialise(self):
        """Clean dataset, initialise context and word weight matrix. Builds word2id and vocab
        dictionary"""
        sampling_size = 0
        word_counter = {}

        for sentence in self.sentences:
            for word in sentence:
                sampling_size += len(sentence)
                word_counter[word] = word_counter.get(word, 0) + 1

        true_word_counter = {}
        for word, counter in word_counter.items():
            if counter >= self.minCount:
                self.w2id.setdefault(word, len(self.w2id))
                true_word_counter[self.w2id[word]] = counter
                self.train_set.append(word)

        print(f'Number words removed : {len(word_counter) - len(true_word_counter)} on '
              f'{len(word_counter)} words')

        nb_unique_words = len(true_word_counter)
        unique_words_idx = list(range(nb_unique_words))

        occ = np.array(list(map(lambda w: true_word_counter[w]**self.alpha, unique_words_idx)))
        prob = occ / np.sum(occ)
        self.sampling_matrix = np.random.choice(unique_words_idx,
                                                size=(sampling_size, self.negativeRate),
                                                p=prob)
        self.sampling_idx = 0

        self.W = np.random.uniform(-1, 1, (nb_unique_words, self.nEmbed))
        self.C = np.random.uniform(-1, 1, (nb_unique_words, self.nEmbed))

        if self.mode_OOV:
            full_vocab = self.train_set.copy()  # will store words + subwords

            # get subwords
            for word in self.train_set:
                word_and_subs = list()  # store list of subwords for each word
                word_and_subs.append(word)
                for idx in range(len(word)):
                    if (idx + self.n_gram) <= len(word):
                        subword = word[idx:(idx + self.n_gram)]
                        word_and_subs.append(subword)
                        if subword not in full_vocab:
                            full_vocab.append(subword)  # append to full vocab

                # index here are the same as in trainset
                self.words_and_their_ngrams.append(word_and_subs)

            N = len(self.train_set)
            N_full = len(full_vocab)

            # indexing for W matrice
            self.n_gram_w2id = {w: idx for (idx, w) in enumerate(full_vocab)}

            # W includes words and subwords
            self.W = np.random.uniform(-1, 1, (N_full, self.nEmbed))
            # C only includes words
            self.C = np.random.uniform(-1, 1, (N, self.nEmbed))

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        # Returns list of 5 negativeIds if negativeRate = 5
        negativeIds = []
        sample_set = self.sampling_matrix[self.sampling_idx]
        self.sampling_idx += 1

        # Verify the Ids are not the context of word id
        for wordid in sample_set:
            if wordid in omit:
                return self.sample(omit)
            else:
                negativeIds.append(wordid)
        return negativeIds

    def train(self):
        self.initialise()
        for counter, sentence in enumerate(self.sentences):
            for wpos, word in enumerate(sentence):
                if word in self.train_set:
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))

                    for context_word in sentence[start:end]:
                        if context_word in self.train_set:
                            ctxtId = self.w2id[context_word]
                            if ctxtId == wIdx:
                                continue
                            negativeIds = self.sample({wIdx, ctxtId})
                            if self.mode_OOV:
                                self.trainWord_with_OOV(wIdx, ctxtId, negativeIds)
                            else:
                                self.trainWord(wIdx, ctxtId, negativeIds)
                            self.trainWords += 1

            if counter % 1000 == 0:
                print(' > training %d of %d' % (counter, len(self.sentences)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.

        # Empty memory after trainings
        self.sentences = []
        del self.sampling_matrix

    def trainWord(self, wIdx, ctxId, negativeIds):
        # Forward propagation
        positive_loss = np.log(1 / (1 + np.exp(-np.dot(self.W[wIdx], self.C[ctxId]))))
        negative_loss = 0
        for i in negativeIds:
            negative_loss += expit(-np.dot(self.W[wIdx], self.C[i]))  # remove - sign
        self.accLoss -= (negative_loss + positive_loss)

        w_prev = self.W[wIdx].copy()
        ctx_prev = self.C[ctxId].copy()
        negative_ctx_prev = [self.C[i].copy() for i in negativeIds]

        # Backward propagation
        self.C[ctxId] -= self.l_r*(expit(np.dot(w_prev, ctx_prev))-1)*w_prev

        for negative_ctx, i in zip(negative_ctx_prev, negativeIds):
            self.C[i] -= self.l_r*(expit(np.dot(w_prev, negative_ctx)))*w_prev

        self.W[wIdx] -= self.l_r*(expit(np.dot(w_prev, ctx_prev))-1)*ctx_prev

        for negative_ctx, i in zip(negative_ctx_prev, negativeIds):
            self.W[wIdx] += self.l_r*(expit(np.dot(w_prev, negative_ctx)))*negative_ctx

    def trainWord_with_OOV(self, wIdx, ctxId, negativeIds):
        """
        Given a words' id, context word id, negative ids
        Compute the word's representation as sum of n-grams + word
        Compute the LCe
        Backpropagate the gradient w.r.t Cpos, Cneg, subword1,2...
        """

        # Forward propagation
        summed_rep_vector = 0.0
        # we want representations of subwords for each word
        for subword in self.words_and_their_ngrams[wIdx]:
            subword_id = self.n_gram_w2id[subword]
            summed_rep_vector += self.W[subword_id]

        # compute gradients
        cpos_word = np.dot(summed_rep_vector, self.C[ctxId])
        # w.r.t to Cpos
        dLce_dCpos = np.dot((expit(cpos_word) - 1), summed_rep_vector)
        # update
        self.C[ctxId] -= self.l_r * dLce_dCpos

        # w.r.t to Cneg_i (and word for computation)
        dLce_dWord_2 = 0
        Lce_2 = 0
        for i in negativeIds:
            cneg_i_word = np.dot(summed_rep_vector, self.C[i])
            dLce_dCneg_i = np.dot(expit(cneg_i_word), summed_rep_vector)
            # update
            self.C[i] -= self.l_r * dLce_dCneg_i
            # keeping these in memory for dword and full loss
            Lce_2 += np.log(expit(np.dot(-self.C[i], summed_rep_vector)))
            dLce_dWord_2 += np.dot(expit(cneg_i_word), self.C[i])

            # w.r.t to word (gradient is the same for all subwords)
        dLce_dWord_1 = np.dot((expit(cpos_word) - 1), self.C[ctxId])
        dLce_dWord = dLce_dWord_1 + dLce_dWord_2
        for subword in self.words_and_their_ngrams[wIdx]:
            subword_id = self.n_gram_w2id[subword]
            self.W[subword_id] -= self.l_r * dLce_dWord

        # full loss
        # if needed to track (across epochs for example)
        # Lce = - np.log(expit(cpos_word)) + Lce_2

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    def calculate_word_vector_OOV(self, word):
        word_vector = np.zeros(self.nEmbed)
        for idx in range(len(word)):
            if (idx + self.n_gram) <= len(word):
                subword = word[idx:(idx + self.n_gram)]

                # look up subword
                if subword in list(self.w2id.keys()):
                    id = self.w2id[subword]
                    sub_vector = self.W[id]
                    word_vector += sub_vector

        # no subword found
        if np.linalg.norm(word_vector) == 0:
            word_vector = self.unknown_vector

        return word_vector

    def similarity(self, word1, word2, add_embed=False):
        """
        Computes similiarity between the two words. Both known and Unknown words'
        embedings are computed as the sum of their subwords embeddings and themselves.
        :param: word1
        :param: word2
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if self.mode_OOV:
            word_vector1 = self.calculate_word_vector_OOV(word1)
            word_vector2 = self.calculate_word_vector_OOV(word2)
        else:
            if word1 in list(self.w2id.keys()):
                id1 = self.w2id[word1]
                word_vector1 = self.W[id1]
                if add_embed:
                    word_vector1 += self.C[id1]
            else:
                word_vector1 = self.unknown_vector

            if word2 in list(self.w2id.keys()):
                id2 = self.w2id[word2]
                word_vector2 = self.W[id2]
                if add_embed:
                    word_vector2 += self.C[id2]
            else:
                word_vector2 = self.unknown_vector

        cosine_sim_num = np.dot(word_vector1, word_vector2)
        cosine_sim_den = np.linalg.norm(word_vector1) * np.linalg.norm(word_vector2)
        cosine_sim = cosine_sim_num / cosine_sim_den

        return cosine_sim

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))
    
    def compute_score(self, testdata):
        df = pd.read_csv(testdata, delimiter='\t')
        M = df['similarity'].max()
        m = df['similarity'].min()
        df['similarity_norm'] = (df['similarity'] - m) / (M - m)
        similarity_predicted, similarity_true = [], []
        for _, row in df.iterrows():
            similarity_predicted.append(self.similarity(row.word1, row.word2))
            similarity_true.append(row.similarity_norm)
        rmse = np.sqrt(np.mean(np.square(np.array(similarity_predicted) -
                                         np.array(similarity_true))))
        return rmse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)',
                        required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences, winSize=10)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print("{:.3f}".format(sg.similarity(a, b, add_embed=True)))
        print(sg.compute_score('simlex.csv'))
