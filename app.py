from flask import Flask
from app.routes import bp  # Import Blueprint

def create_app():
    app = Flask(__name__, template_folder="app/templates")
    app.register_blueprint(bp)  # ✅ Register the Blueprint
    return app

if __name__ == "__main__":
    app = create_app()# ✅ Fix: Use "app.url_map" instead of "bp.url_map"
    app.run(debug=True)
