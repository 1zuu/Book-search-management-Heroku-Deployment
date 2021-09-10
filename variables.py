max_length = 140
trunc_type = 'post'
padding = 'post'
pad_token = '<pad>'
oov_tok = "<oov>"

model_weights = 'data/feature_model_weights.h5'
tflite_weights = 'data/feature_model_weights.tflite'
encoder_path = 'data/label_encoder.pickle'
tokenizer_weights = 'data/tokenizer.pickle'
data_path = 'data/books.csv'
json_dir = 'data/prices/'

seed = 42
heroku_url = 'https://book-search-management.herokuapp.com/books'
host = '0.0.0.0'
port = 5000
embedding_dim = 150
num_epochs = 20
batch_size = 32
size_lstm  = 256
dense1 = 512
dense2 = 256
dense3 = 128
dense4 = 64
rate = 0.2
learning_rate = 0.001

test_size = 0.25
n_matches = 5