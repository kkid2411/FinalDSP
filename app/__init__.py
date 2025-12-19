import os
from flask import Flask


def _load_config(app: Flask):
    config_name = os.getenv("FLASK_CONFIG", "DevConfig")
    try:
        app.config.from_object(f"config.{config_name}")
    except (ImportError, AttributeError):
        app.config.from_object("config.DevConfig")

    upload_subdir = app.config.get("UPLOAD_SUBDIR", "uploads")
    upload_folder = os.path.join(app.instance_path, upload_subdir)
    app.config["UPLOAD_FOLDER"] = upload_folder

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(upload_folder, exist_ok=True)


def create_app():
    """Application factory."""
    app = Flask(
        __name__,
        instance_relative_config=True,
        static_folder="static",
        template_folder="templates",
    )

    _load_config(app)

    from .routes import main_bp

    app.register_blueprint(main_bp)

    return app
