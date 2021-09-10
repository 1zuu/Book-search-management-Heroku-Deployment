from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from heroku_inference import BSM_Heroku_Inference

from variables import*

app = Flask(__name__)
CORS(app)

model = BSM_Heroku_Inference()
model.run()

@app.route("/books", methods=['GET','POST'])
@cross_origin()
def predictions():
    try:
        book_description = request.get_json(force=True)
        response = model.predict_book(book_description)
        return jsonify(response)

    except Exception as e:
        print(e)

if __name__ == '__main__':
    app.run(debug=True, host=host, port=port, threaded=False, use_reloader=False)