

import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database
# print(asl.df.head())

words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit


features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']

df_means = asl.df.groupby('speaker').mean()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
from asl_utils import test_std_tryit
df_std = asl.df.groupby('speaker').std()
cols = ['right-x', 'right-y', 'left-x', 'left-y']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

for i, feature in enumerate(features_norm):
    col = cols[i]
    means = asl.df['speaker'].map(df_means[col])
    stds = asl.df['speaker'].map(df_std[col])
    asl.df[feature] = (asl.df[col] - means ) / stds

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

x = asl.df['grnd-rx']
y = asl.df['grnd-ry']
x2 = np.power(x,2)
y2 = np.power(y,2)
r = np.power(x2 + y2, 0.5)
asl.df['polar-rtheta'] = np.arctan2(x,y)
asl.df['polar-rr'] = r

x = asl.df['grnd-lx']
y = asl.df['grnd-ly']
x2 = np.power(x,2)
y2 = np.power(y,2)
r = np.power(x2 + y2, 0.5)
asl.df['polar-ltheta'] = np.arctan2(x,y)
asl.df['polar-lr'] = r

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

asl.df['delta-rx'] = asl.df['grnd-rx'].diff()
asl.df['delta-ry'] = asl.df['grnd-ry'].diff()
asl.df['delta-lx'] = asl.df['grnd-lx'].diff()
asl.df['delta-ly'] = asl.df['grnd-ly'].diff()

asl.df.fillna(method='backfill', inplace=True)

features_custom = ['speed-r', 'speed-l']

dx2 = np.power(asl.df['delta-rx'],2)
dy2 = np.power(asl.df['delta-ry'],2)
speed = np.power(dx2 + dy2, 1/2)/1
asl.df['speed-r'] = speed

dx2 = np.power(asl.df['delta-lx'],2)
dy2 = np.power(asl.df['delta-ly'],2)
speed = np.power(dx2 + dy2, 1/2)/1
asl.df['speed-l'] = speed




# ===========================================================================================================

if 0:
    from my_model_selectors import SelectorCV

    training = asl.build_training(features_polar)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    for word in words_to_train:
        start = timeit.default_timer()
        model = SelectorCV(sequences, Xlengths, word, 
                        min_n_components=2, max_n_components=15, random_state = 14).select()
        end = timeit.default_timer()-start
        if model is not None:
            print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
        else:
            print("Training failed for {}".format(word))

# print('-'*30)
if 0:
    # TODO: Implement SelectorBIC in module my_model_selectors.py
    from my_model_selectors import SelectorBIC

    training = asl.build_training(features_polar)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    for word in words_to_train:
        start = timeit.default_timer()
        model = SelectorBIC(sequences, Xlengths, word, 
                        min_n_components=2, max_n_components=15, random_state = 14).select()
        end = timeit.default_timer()-start
        if model is not None:
            print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
        else:
            print("Training failed for {}".format(word))

# print('-'*30)
if 0:
    # TODO: Implement SelectorDIC in module my_model_selectors.py
    from my_model_selectors import SelectorDIC

    training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    for word in words_to_train:
        start = timeit.default_timer()
        model = SelectorDIC(sequences, Xlengths, word, 
                        min_n_components=2, max_n_components=15, random_state = 14).select()
        end = timeit.default_timer()-start
        if model is not None:
            print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
        else:
            print("Training failed for {}".format(word))


# ===========================================================================================================


from my_recognizer import recognize
from asl_utils import show_errors
from my_model_selectors import SelectorConstant, SelectorBIC, SelectorCV, SelectorDIC
from asl_data import SinglesData, WordsData

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

def my_show_errors(guesses: list, test_set: SinglesData):
    """ Print WER and sentence differences in tabular form

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        nothing returned, prints error report

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1
    
    wer = float(S) / float(N)
    # print("\n**** WER = {}".format(float(S) / float(N)))
    # print("Total correct: {} out of {}".format(N - S, N))
    return wer




all_features = {'features_ground':features_ground,
                'features_norm':features_norm,
                'features_polar':features_polar,
                'features_delta':features_delta,
                'features_custom':features_custom}

all_selectors = {'SelectorConstant':SelectorConstant, 
                'SelectorBIC':SelectorBIC, 
                'SelectorCV':SelectorCV, 
                'SelectorDIC':SelectorDIC}

if 0:
    features = features_ground # change as needed
    model_selector = SelectorConstant # change as needed

    models = train_all_words(features, model_selector)
    test_set = asl.build_test(features)
    probabilities, guesses = recognize(models, test_set)
    my_show_errors(guesses, test_set)

print('='*30)

for features_name, features in all_features.items():
    for model_selector_name, model_selector in all_selectors.items():
        models = train_all_words(features, model_selector)
        test_set = asl.build_test(features)
        probabilities, guesses = recognize(models, test_set)
        wer = my_show_errors(guesses, test_set)
        print('{} - {} - {}'.format(features_name, model_selector_name, wer))

print('='*30)