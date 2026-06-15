# pyrefly: ignore [missing-import]
from schemas.prediction_schema import validateContent
from handlers.predict_handler import predictHandler
# pyrefly: ignore [missing-import]
from flask import Blueprint, request, jsonify
# pyrefly: ignore [missing-import]
from pydantic import ValidationError


prediction_bp = Blueprint(
    "prediction",
    __name__
)

@prediction_bp.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        body = request.get_json()

        validated_data = validateContent(body)
        result = predictHandler(validated_data)
        return jsonify(result)

    except ValidationError as e:

        return jsonify({
            "error": e.errors()
        }), 400

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500