from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from heroku_inference import BSM_Heroku_Inference

from variables import*

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

model = BSM_Heroku_Inference()
model.run()

@app.route("/<filter>", methods=['GET','POST'])
@cross_origin()
def predictions(filter):
    book_description = request.get_json(force=True)
    response = model.predict_book(book_description, filter)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host=host, port=port, threaded=False, use_reloader=False)