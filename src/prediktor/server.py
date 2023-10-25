import flask

from prediktor import infilling

app = flask.Flask(__name__)


@app.route("/status/")
def get_status():
    return "<p>It works!</p>"


@app.route("/", methods=["POST"])
def submit_text_for_prediction():
    request = flask.request.get_json()
    text = request["text"]
    cursor_pos = request["cursor"]
    prediction = infilling.infill(text, cursor_pos)
    return flask.jsonify({"prediction": prediction})
