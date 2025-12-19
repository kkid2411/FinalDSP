import os


class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 32 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}
    UPLOAD_SUBDIR = os.getenv("UPLOAD_SUBDIR", "uploads")


class DevConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False
