from typing import Any

import flask

from prediktor import infilling, prediction

app = flask.Flask(__name__)


@app.route("/status/")
def get_status() -> str:
    return "<p>It works!</p>"


@app.route("/", methods=["POST"])
def submit_text_for_prediction() -> flask.Response:
    request: Any = flask.request.get_json()
    text: str = request["text"]
    cursor_pos: int = request["cursor"]
    if cursor_pos >= len(text.rstrip()):
        output = prediction.confidence_predictor.predict(text)
    else:
        output = infilling.blank_infiller.infill(text, cursor_pos)
    return flask.jsonify({"prediction": output})
