from flask import Flask, jsonify

from api.create_model_api import create_model_bp
from api.predict_api import predict_bp
from api.signal_processing_api import signal_processing_bp
app = Flask(__name__)

app.register_blueprint(signal_processing_bp)
app.register_blueprint(create_model_bp)

app.register_blueprint(predict_bp)
@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!????????'


if __name__ == '__main__':
    app.run(debug=True)

