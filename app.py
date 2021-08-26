from heroku_inference import BSM_Heroku_Inference
from flask import Flask, Response, request
from variables import*

app = Flask(__name__)


model = BSM_Heroku_Inference()
model.run()

@app.route("/books", methods=["POST"])
def predictions():
    try:
        book_description = request.get_json()
        response = model.predict_book(book_description)
        # return Response(
        #             response=response, 
        #             status=500, 
        #             mimetype="application/json"
        #             )
        return response

    except Exception as e:
        print(e)

if __name__ == '__main__':
    app.run(debug=True, host=heroku_host, port=heroku_port, threaded=False, use_reloader=False)


# {"description" : "In the highly anticipated Thinking, Fast and Slow, Kahneman takes us on a groundbreaking tour of the mind and explains the two systems that drive the way we think. System 1 is fast, intuitive, and emotional; System 2 is slower, more deliberative, and more logical. Kahneman exposes the extraordinary capabilities—and also the faults and biases—of fast thinking, and reveals the pervasive influence of intuitive impressions on our thoughts and behavior. The impact of loss aversion and overconfidence on corporate strategies, the difficulties of predicting what will make us happy in the future, the challenges of properly framing risks at work and at home, the profound effect of cognitive biases on everything from playing the stock market to planning the next vacation—each of these can be understood only by knowing how the two systems work together to shape our judgments and decisions."}   
