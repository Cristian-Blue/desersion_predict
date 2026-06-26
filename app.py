# pyrefly: ignore [missing-import]
from flask import Flask
from flask_cors import CORS

from routes.prediction_routes import prediction_bp

app = Flask(__name__)

CORS(app)

app.register_blueprint(
    prediction_bp,
    url_prefix="/api/v1"
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)