import pandas as pd
import numpy as np

class SpaceGen_OneHotVectorizer:
    @staticmethod
    def string_vectorizer(string, vocabulary, max_len):
      '''
      Returns a one hot encoded vector for a given string,
      vocabulary and desirable max_len representation of words
      '''
      empty = SpaceGen_OneHotVectorizer.empty_matrix(max_len, vocabulary)
      empty[0, vocabulary.index(string)] = 1
      return empty

    @staticmethod
    def create_vocabulary(list_of_words):
      '''
      Returns the vocabulary of a given list of words
      '''
      vocabulary = set(''.join(list_of_words))
      vocabulary = sorted([i for i in vocabulary])
      return vocabulary

    @staticmethod
    def empty_matrix(max_len, vocabulary):
      '''
      Returns an empty matrix for a given max_len and vocabulary
      '''
      array = []
      for i in range(max_len):
        array.append([0] * len(vocabulary))
      return np.array(array)

    def X_to_one_hot_matrix(X, vocabulary, max_len):
      '''
      For a given X with past and future and vocaulary, returns a matrix representation for each sequence (row)
      '''
      df = pd.DataFrame(X)
      df_mat = df.applymap(lambda x: SpaceGen_OneHotVectorizer.string_vectorizer(x, vocabulary = vocabulary, max_len = 1))
      df_mat['matrix_representation'] = df_mat.apply(lambda x: list(x), axis=1)
      df_mat['matrix_representation'] = df_mat['matrix_representation'].apply(lambda x: np.stack(x))
      df_mat['matrix_representation'] = df_mat['matrix_representation'].apply(lambda x: np.vstack(x))
      return df_mat['matrix_representation']