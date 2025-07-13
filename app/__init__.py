from flask import Flask

def create_app():
    app = Flask(__name__)

    with app.app_context():
        from .routes import bp
        app.register_blueprint(bp)

    return app
