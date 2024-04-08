from typing import Any

import flask

from preditor.config import Config
from preditor.infilling import infilling
from preditor.model.hf import HFModel
from preditor.prediction import prediction

app = flask.Flask(__name__)
model = HFModel(Config.model_path)


@app.route("/status/")
def get_status() -> str:
    return "<p>It works!</p>"


@app.route("/", methods=["POST"])
def submit_text_for_prediction() -> flask.Response:
    request: Any = flask.request.get_json()
    text: str = request["text"]
    cursor_pos: int = request["cursor"]
    if cursor_pos >= len(text.rstrip()):
        output = prediction.predict(model, text)
    else:
        output = infilling.infill(model, text, cursor_pos)
    return flask.jsonify({"prediction": output})
