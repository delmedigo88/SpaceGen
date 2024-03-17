import pandas as pd
import numpy as np

class SpaceGen_preprocessing:
  def __init__(self, content = "helloworld", size= 10, past_capacity = 5 , future_capacity = 5):
    self.size = size
    self.content = content[:self.size]
    self.past_capacity = past_capacity
    self.future_capacity = future_capacity
    self.num_features = self.past_capacity + self.future_capacity + 1 # 1 for letter
    self.vocabulary = []

  def create_vocabulary(self, correct_txt):
    '''
    Returns the unique letters of the given text + '-1'
    '''
    vocabulary = list({b for b in bytes(correct_txt, 'utf-8')})
    vocabulary.append(-1)
    vocabulary = sorted(vocabulary)
    self.vocabulary = vocabulary
    return None

  @staticmethod
  def create_decision_vector(W: list, C: list):
    '''
    Returns the Decision Vector(D),
    given Wrong Vector(W) and Correct Vector(C)
    '''
    D = []
    w_i = 0
    c_i = 0
    while w_i < len(W):
      if W[w_i] == C[c_i]:
          D.append('K')
          w_i += 1
          c_i += 1
      elif W[w_i] == 32 and C[c_i] != 32 :
          D.append('D')
          w_i += 1
      elif C[c_i] == 32 and W[w_i] != 32:
          D.append('I')
          c_i += 1
          w_i += 1
      else:
          c_i += 1
    return D


  @staticmethod
  def to_correct(W, D):
      '''
      Returns the correct text,
      given Wrong Vector(W) and Decision Vector(D)
      '''
      output_vec = []
      for i in range(0, len(D)):
        if D[i] == 'K':
          output_vec.append(W[i])
        elif D[i] == 'I':
          output_vec.append(32)
          output_vec.append(W[i])
        elif D[i] == 'D':
          pass
      decoded_text = bytes(output_vec).decode()
      return decoded_text


  @staticmethod
  def to_bytes_list(text: str, encoding = 'UTF-8'):
      '''
      Returns the bytes list of a given text
      '''
      return [b for b in bytes(text, encoding)]


  @staticmethod
  def to_one_hot_df(wrong_txt, D):
    '''
    Returns the one hot encoded dataframe,
    given Wrong Vector(W) and Decision Vector(D)
    '''
    df = pd.DataFrame({'letter':[l for l in wrong_txt],'decision':D})
    encoding =  OneHotEncoder()
    y_matrix =  encoding.fit_transform(df[['decision']])
    onehot_df = pd.DataFrame(y_matrix.toarray(), columns = encoding.get_feature_names_out(['decision']) )
    onehot_df = onehot_df.astype('int')
    example_df = pd.concat([df, onehot_df], axis=1)
    example_df =example_df.drop(['decision'], axis=1)
    return example_df


  @staticmethod
  def decode_vec(arr):
    '''
    Returns the decoded text,
    given the bytes list
    '''
    return bytes(arr).decode()


  @staticmethod
  def sliding_window_past(arr, window_size = 5):
    '''
    Returns the past sliding window of the given array and window size
    '''
    arr = list(arr)
    new_arr = []
    for i in range(len(arr)):
      start_window = max(0, i- window_size)
      tmp_seq = arr[start_window:i]
      if window_size - len(tmp_seq) ==0:
        new_arr.append(tmp_seq)
      else:
        new_arr.append([-1] * (window_size - len(tmp_seq)) + tmp_seq)
    return new_arr


  @staticmethod
  def sliding_window_future(arr, window_size = 5):
    '''
    Returns the future sliding window of the given array and window size
    '''
    arr = list(arr)
    seq = []
    for i in range(len(arr)):
      p = arr[i+1:i+window_size+1]
      if window_size - len(p) ==0:
        seq.append(p)
      else:
        seq.append(p + [-1] * (window_size - len(p)))
    return seq

  @staticmethod
  def insert_random_spaces(text, percent = .25):
    '''
    Returns the text with random spaces inserted
    '''
    l = list(text)
    rand_indices = np.random.randint(0, len(l)+1, int(np.round(len(l) * percent)))
    print(rand_indices)
    t = 1
    for i in range(len(l)+1):
      if i in rand_indices:
          l.insert(i + t, ' ')
          t+=1
    new_txt = ''.join(l).strip()
    return new_txt


  @staticmethod
  def prob_to_decision(a):
    '''
    Return I or K given probability vector
    '''
    if a[0] > a[1]:
      return 'I'
    else:
      return 'K'