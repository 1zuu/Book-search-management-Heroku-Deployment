import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from variables import *
from util import*

np.random.seed(seed)
warnings.simplefilter("ignore", DeprecationWarning)

class BSM_Heroku_Inference(object):
    def __init__(self):
        df_response, prices = get_Data()
        self.df_response = df_response
        self.prices = prices

    def TFinterpreter(self):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=tflite_weights)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def TFliteInference(self, input_data):
        input_shape = self.input_details[0]['shape']
        if len(input_shape) == 1:
            input_data = np.expand_dims(input_shape, axis=0).astype(np.float32)
        assert np.array_equal(input_shape, input_data.shape), "required shape : {} doesn't match with provided shape : {}".format(input_shape, input_data.shape)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        concern_type = self.interpreter.get_tensor(self.output_details[0]['index']) #Concern_Type
        department = self.interpreter.get_tensor(self.output_details[1]['index']) #Department
            
        return concern_type, department

    def get_category_from_label(self, category):
        with open(encoder_path, 'rb') as handle:
            encoder = pickle.load(handle)
        category = encoder.inverse_transform([category]).squeeze()
        category = str(category).lower()
        return category

    def predict_category(self, description):
        processed_description = prediction_data(description).squeeze()
        category, feature = self.TFliteInference(processed_description.reshape(1,max_length).astype(np.float32))

        category = np.argmax(category)
        category = self.get_category_from_label(category)
        return category, feature
        
    def predict_features(self, category):
        df_response, price_response = load_category_df(self.df_response, self.prices, category)
        df_response = df_response.reset_index(drop=True)
        category_descriptions = df_response['Description'].values
        processed_descriptions = preprocessed_data(category_descriptions)

        features = np.array(
                    [self.TFliteInference(prediction_data(x).reshape(1,max_length).astype(np.float32))[1] for x in processed_descriptions])

        return features, df_response, price_response

    def predict_book(self,request):
        description = request['description']
        category, pred_feature = self.predict_category(description)
        features, df_response, price_response = self.predict_features(category)
        cos_sim = {i:float(cosine_similarity(pred_feature, feature).squeeze()) for i,feature in enumerate(features)}
        cos_sim = dict(sorted(cos_sim.items(), key=lambda item: item[1], reverse=True))
        top_matches = list(cos_sim.keys())[:n_matches]
        df_top_match = df_response.iloc[top_matches]
        price_top_match = price_response[top_matches]

        df_top_match = df_top_match[['Book_title' ,'Author' ,'ISBN-10' ,'ISBN-13', 'Cover_link']]
        
        
        response = {}
        books = {}

        response_data = []

        for i in range(n_matches):
            book = {}
            book['title'] = df_top_match.iloc[i]['Book_title'] if df_top_match.iloc[i]['Book_title'] else None
            book['author'] = df_top_match.iloc[i]['Author'] if df_top_match.iloc[i]['Author'] else None
            book['isbn_10'] = df_top_match.iloc[i]['ISBN-10'] if df_top_match.iloc[i]['ISBN-10'] else None
            book['isbn_13'] = df_top_match.iloc[i]['ISBN-13'] if df_top_match.iloc[i]['ISBN-13'] else None
            book['cover_photo'] = df_top_match.iloc[i]['Cover_link'] if df_top_match.iloc[i]['Cover_link'] else None
            book['websites'] = reform_prices(price_top_match[i])
            books['{}'.format(i+1)] = book
        response_data.append(books)
        response_data.append({"category" : category})
        response_data.append({"type" : "book"})

        response['data'] = response_data
        return response

    def run(self):
        self.TFinterpreter()