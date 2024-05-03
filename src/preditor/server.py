"""This module is the entry point for the application.

It provides a REST API for the individual tasks.
"""

import abc
from typing import Any, Type

import flask
import pydantic

from preditor.config import Config
from preditor.infilling import infilling
from preditor.model.hf import HFModel
from preditor.prediction import prediction
from preditor.substitution import substitution
from preditor.suggestion import suggestion

app = flask.Flask(__name__)
model = HFModel(Config.model_path)


class PreditorRequest(pydantic.BaseModel, abc.ABC):
    """Interface for a request to the Preditor API."""

    @abc.abstractmethod
    def handle(self) -> str:
        """Handle the request and return the output."""
        pass


class SuggestionRequest(PreditorRequest):
    """Request for a suggestion.

    It combines the prediction and infilling tasks.
    """

    before_cursor: str
    after_cursor: str
    prediction_config: prediction.PredictionConfig = prediction.PredictionConfig()
    infilling_config: infilling.InfillingConfig = infilling.InfillingConfig()

    def handle(self) -> str:
        return suggestion.suggest(
            model, self.before_cursor, self.after_cursor,
            self.prediction_config, self.infilling_config
        )


class SubstitutionRequest(PreditorRequest):
    """Request for a substitution of a word in a sentence."""

    before_old: str
    old: str
    after_old: str
    replacement: str
    config: substitution.SubstitutionConfig = substitution.SubstitutionConfig()

    def handle(self) -> str:
        return substitution.replace(
            model,
            self.before_old, self.old, self.after_old, self.replacement,
            self.config
        )


@app.route("/")
def get_status() -> str:
    """Show that the server is running."""
    return "<h1>Preditor</h1>"


@app.route("/suggest/", methods=["POST"])
def suggest() -> flask.Response:
    """Dispatch a suggestion request."""
    request: Any = flask.request.get_json()
    return _handle_request(request, SuggestionRequest)


@app.route("/substitute/", methods=["POST"])
def substitute() -> flask.Response:
    """Dispatch a substitution request."""
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
