import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        states_range = range(self.min_n_components, self.max_n_components + 1)

        best_score = float("inf")
        best_model = None

        n_frs = self.X.shape[1]
        logN = math.log( sum(self.lengths))
        for n_states in states_range:
            try:
                model = GaussianHMM(n_components=n_states, n_iter=1000, random_state=self.random_state, verbose=False)
                fit_model = model.fit(self.X, self.lengths)
                logl = fit_model.score(self.X, self.lengths)
            except:
                continue
            n = n_states
            p = (n*n) + 2*n_frs*n - 1
            bic = -2 * logl + logN * p

            if bic < best_score:
                best_score = bic
                n_components = n_states
                best_model = fit_model     

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score = float("-Inf")
            best_model = None
            states_range = range(self.min_n_components, self.max_n_components + 1)

            all_words = self.words.keys()

            for n_components in states_range:
                hmm_model = self.base_model(n_components)
                scores= []
                for word in all_words:
                    if (word != self.this_word):
                        word_X, word_lengths = self.all_word_Xlengths[word]
                        scores.append(hmm_model(word_X, word_lengths))
                score = hmm_model.score(self.X, self.lengths) - np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_model = hmm_model
            return best_model
        except:
            return self.base_model(self.n_constant)
        '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = -float("inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components + 1):
            score_other = 0
            try:
                model = self.base_model(n)
                score = model.score(self.X, self.lengths)
                other_words_copy = self.words.copy()
                del other_words_copy[self.this_word]
                for word in other_words_copy:
                    otherX, otherlength = self.hwords[word]
                    try:
                        score_other = score_other + model.score(otherX, otherlength)
                    except:
                        pass
                score_other = score_other / len(other_words_copy)
                DIC_score = score - score_other
                if DIC_score > best_score:
                    best_score = DIC_score
                    best_model = model
            except:
                pass
        return best_model    
        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = -float("inf")
        best_num_state = 1
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            i = 0
            total_score = 0
            word_sequences = self.sequences
            try:
                split_method = KFold(n_splits=min(3, len(word_sequences)))
            except:
                return None
            try:
                for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                    cv_train_x, cv_train_length = combine_sequences(cv_train_idx, word_sequences)
                    cv_test_x, cv_test_length = combine_sequences(cv_test_idx, word_sequences)
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(cv_train_x, cv_train_length)
                    total_score = total_score + model.score(cv_test_x, cv_test_length)
                    i = i + 1
                if total_score / i > best_score:
                    best_score = total_score / i
                    best_num_state = n
            except:
                pass
        best_model = self.base_model(best_num_state)
        return best_model
    


