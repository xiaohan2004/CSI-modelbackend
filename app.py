from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from api.csi_model_api import csi_model_bp
from api.model_api import model_api_bp

app = Flask(__name__)
# 配置CORS，允许所有源访问API
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")
app.register_blueprint(csi_model_bp)

app.register_blueprint(model_api_bp)


@app.route("/")
def hello_world():  # put application's code here
    return "Hello World!????????"


if __name__ == "__main__":
    app.run(debug=True)
