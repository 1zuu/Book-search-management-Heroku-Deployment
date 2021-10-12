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
        # category = self.feature_model.predict(prediction_data(description))[0][0]
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

    # def loaded_model(self): # Load and compile pretrained model
    #     self.feature_model = load_model(model_weights)
    #     self.feature_model.compile(
    #                     loss='sparse_categorical_crossentropy', 
    #                     optimizer=Adam(lr=learning_rate), 
    #                     metrics=['accuracy']
    #                     )

    def predict_book(self,request, filter):
        if filter == 'description':
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

        elif filter == 'isbn_10':
            response = {}
            self.df_response['ISBN-10'] = self.df_response['ISBN-10'].astype(str)
            isbn_10 = request['isbn_10']
            matches = self.df_response.index[self.df_response['ISBN-10'] == isbn_10].tolist()
            df_response = self.df_response.loc[matches, :]

            if len(matches) != 1:
                response['data'] = []
                return response
            else:
                idx = matches[0]
                price_json = self.prices[idx]

                response_data = []

                book = {}
                books = {}
                book['title'] = df_response['Book_title'].values[0]
                book['author'] = df_response['Author'].values[0]
                book['isbn_10'] = df_response['ISBN-10'].values[0]
                book['isbn_13'] = df_response['ISBN-13'].values[0]
                book['cover_photo'] = df_response['Cover_link'].values[0]
                book['websites'] = reform_prices(price_json)
                books['{}'.format(1)] = book
                response_data.append(books)
                response_data.append({"category" : df_response['Category'].values[0]})
                response_data.append({"type" : "book"})

                response['data'] = response_data
                return response

        elif filter == 'isbn_13':
            response = {}
            self.df_response['ISBN-13'] = self.df_response['ISBN-13'].astype(str)
            isbn_13 = request['isbn_13']
            matches = self.df_response.index[self.df_response['ISBN-13'] == isbn_13].tolist()
            df_response = self.df_response.loc[matches, :]

            if len(matches) != 1:
                response['data'] = []
                return response
            else:
                idx = matches[0]
                price_json = self.prices[idx]

                response_data = []

                book = {}
                books = {}
                book['title'] = df_response['Book_title'].values[0]
                book['author'] = df_response['Author'].values[0]
                book['isbn_10'] = df_response['ISBN-10'].values[0]
                book['isbn_13'] = df_response['ISBN-13'].values[0]
                book['cover_photo'] = df_response['Cover_link'].values[0]
                book['websites'] = reform_prices(price_json)
                books['{}'.format(1)] = book
                response_data.append(books)
                response_data.append({"category" : df_response['Category'].values[0]})
                response_data.append({"type" : "book"})

                response['data'] = response_data
                return response

        elif filter == 'title':
            response = {}
            self.df_response['Book_title'] = self.df_response['Book_title'].astype(str)
            self.df_response['Book_title'] = self.df_response['Book_title'].str.lower()

            title = request['title'].lower()
            matches = self.df_response.index[self.df_response['Book_title'] == title].tolist()
            df_response = self.df_response.loc[matches, :]

            if len(matches) != 1:
                response['data'] = []
                return response
            else:
                idx = matches[0]
                price_json = self.prices[idx]

                response_data = []

                book = {}
                books = {}
                book['title'] = df_response['Book_title'].values[0]
                book['author'] = df_response['Author'].values[0]
                book['isbn_10'] = df_response['ISBN-10'].values[0]
                book['isbn_13'] = df_response['ISBN-13'].values[0]
                book['cover_photo'] = df_response['Cover_link'].values[0]
                book['websites'] = reform_prices(price_json)
                books['{}'.format(1)] = book
                response_data.append(books)
                response_data.append({"category" : df_response['Category'].values[0]})
                response_data.append({"type" : "book"})

                response['data'] = response_data
                return response
        
        elif filter == 'author':
            response = {}
            self.df_response['Author'] = self.df_response['Author'].astype(str)
            self.df_response['Author'] = self.df_response['Author'].str.lower()    

            author = request['author'].lower()
            matches = self.df_response.index[self.df_response['Author'] == author].tolist()
            if len(matches) == 0:
                response['data'] = []
                return response
            else:
                df_response = self.df_response.loc[matches, :]
                price_top_match = self.prices[matches]

                response = {}
                books = {}

                response_data = []

                for i in range(len(matches)):
                    book = {}
                    book['title'] = df_response.iloc[i]['Book_title'] if df_response.iloc[i]['Book_title'] else None
                    book['author'] = df_response.iloc[i]['Author'] if df_response.iloc[i]['Author'] else None
                    book['isbn_10'] = df_response.iloc[i]['ISBN-10'] if df_response.iloc[i]['ISBN-10'] else None
                    book['isbn_13'] = df_response.iloc[i]['ISBN-13'] if df_response.iloc[i]['ISBN-13'] else None
                    book['cover_photo'] = df_response.iloc[i]['Cover_link'] if df_response.iloc[i]['Cover_link'] else None
                    book['websites'] = reform_prices(price_top_match[i])
                    books['{}'.format(i+1)] = book

                response_data.append(books)
                response_data.append({"category" : "NULL"})
                response_data.append({"type" : "book"})

                response['data'] = response_data
                return response

    def run(self):
        # self.loaded_model()
        self.TFinterpreter()