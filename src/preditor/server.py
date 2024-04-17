import abc
from typing import Any, Dict, Type

import flask
import pydantic

from preditor.config import Config
from preditor.infilling import infilling
from preditor.model.hf import HFModel
from preditor.prediction import prediction
from preditor.substitution import substitution

app = flask.Flask(__name__)
model = HFModel(Config.model_path)


class PreditorRequest(pydantic.BaseModel, abc.ABC):
    @abc.abstractmethod
    def handle(self) -> str:
        pass


class InfillingRequest(PreditorRequest):
    text: str
    cursor: int
    config: infilling.InfillingConfig = infilling.InfillingConfig()

    def handle(self) -> str:
        return infilling.infill(model, self.text, self.cursor, self.config)


class PredictionRequest(PreditorRequest):
    text: str
    config: prediction.PredictionConfig = prediction.PredictionConfig()

    def handle(self) -> str:
        return prediction.predict(model, self.text, self.config)


class SubstitutionRequest(PreditorRequest):
    text: str
    start: int
    length: int
    replacement: str
    config: substitution.SubstitutionConfig = substitution.SubstitutionConfig()

    def handle(self) -> str:
        return substitution.replace(
            model,
            self.text, self.start, self.length, self.replacement,
            self.config
        )


@app.route("/")
def get_status() -> str:
    return "<h1>Preditor</h1>"


@app.route("/infill/", methods=["POST"])
def infill() -> flask.Response:
    request: Any = flask.request.get_json()
    return _handle_request(request, InfillingRequest)


@app.route("/predict/", methods=["POST"])
def predict() -> flask.Response:
    request: Any = flask.request.get_json()
    return _handle_request(request, PredictionRequest)


@app.route("/substitute/", methods=["POST"])
def substitute() -> flask.Response:
    data: Any = flask.request.get_json()
    return _handle_request(data, SubstitutionRequest)


def _handle_request(data: Any, request_cls: Type[PreditorRequest]) -> flask.Response:
    """Handle a generic request for one of the tasks."""
    try:
        request = request_cls(**data)
    except pydantic.ValidationError as e:
        details = e.errors(include_input=False, include_url=False)
        return _build_error_response("Invalid request data", details)
    output = request.handle()
    return flask.jsonify({"output": output})


def _build_error_response(msg: str, details: Any = None) -> flask.Response:
    """Create a response with an error code and error description."""
    body = {"error": msg}
    if details:
        body["details"] = details
    response = flask.jsonify(body)
    response.status_code = 400
    return response
