import os

def detect_env():
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "cloud"

ENV = detect_env()

if ENV == "colab":
    DATA_PATH = "/content/drive/MyDrive/209B/"
else:
    DATA_PATH = "~/hl/ac109_project/raw"
