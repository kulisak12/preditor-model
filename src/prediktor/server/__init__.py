import flask

from prediktor.server.prediction import predict


app = flask.Flask(__name__)


@app.route("/status/")
def get_status():
    return "<p>It works!</p>"


@app.route("/", methods=["POST"])
def submit_text_for_prediction():
    request = flask.request.get_json()
    text = request["text"]
    prediction = predict(text)
    return flask.jsonify({"prediction": prediction})
