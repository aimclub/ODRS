# Ultralytics YOLO 🚀, GPL-3.0 license

import requests

from ultralytics.hub.auth import Auth
from ultralytics.hub.session import HubTrainingSession
from ultralytics.hub.utils import split_key
from ultralytics.yolo.engine.exporter import EXPORT_FORMATS_LIST
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import LOGGER, PREFIX, emojis

# Define all export formats
EXPORT_FORMATS_HUB = EXPORT_FORMATS_LIST + ["ultralytics_tflite", "ultralytics_coreml"]


def start(key=""):
    """
    Start training models with Ultralytics HUB. Usage: from src.ultralytics import start; start('API_KEY')
    """
    auth = Auth(key)
    try:
        if not auth.get_state():
            model_id = request_api_key(auth)
        else:
            _, model_id = split_key(key)

        if not model_id:
            raise ConnectionError(emojis('Connecting with global API key is not currently supported. ❌'))

        session = HubTrainingSession(model_id=model_id, auth=auth)
        session.check_disk_space()

        trainer = YOLO(session.input_file)
        session.register_callbacks(trainer)
        trainer.train(**session.train_args)
    except Exception as e:
        LOGGER.warning(f"{PREFIX}{e}")


def request_api_key(auth, max_attempts=3):
    """
    Prompt the user to input their API key. Returns the model ID.
    """
    import getpass
    for attempts in range(max_attempts):
        LOGGER.info(f"{PREFIX}Login. Attempt {attempts + 1} of {max_attempts}")
        input_key = getpass.getpass("Enter your Ultralytics HUB API key:\n")
        auth.api_key, model_id = split_key(input_key)

        if auth.authenticate():
            LOGGER.info(f"{PREFIX}Authenticated ✅")
            return model_id

        LOGGER.warning(f"{PREFIX}Invalid API key ⚠️\n")

    raise ConnectionError(emojis(f"{PREFIX}Failed to authenticate ❌"))


def reset_model(key=""):
    # Reset a trained model to an untrained state
    api_key, model_id = split_key(key)
    r = requests.post("https://api.ultralytics.com/model-reset", json={"apiKey": api_key, "modelId": model_id})

    if r.status_code == 200:
        LOGGER.info(f"{PREFIX}model reset successfully")
        return
    LOGGER.warning(f"{PREFIX}model reset failure {r.status_code} {r.reason}")


def export_model(key="", format="torchscript"):
    # Export a model to all formats
    assert format in EXPORT_FORMATS_HUB, f"Unsupported export format '{format}', valid formats are {EXPORT_FORMATS_HUB}"
    api_key, model_id = split_key(key)
    r = requests.post("https://api.ultralytics.com/export",
                      json={
                          "apiKey": api_key,
                          "modelId": model_id,
                          "format": format})
    assert (r.status_code == 200), f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} export started ✅")


def get_export(key="", format="torchscript"):
    # Get an exported model dictionary with download URL
    assert format in EXPORT_FORMATS_HUB, f"Unsupported export format '{format}', valid formats are {EXPORT_FORMATS_HUB}"
    api_key, model_id = split_key(key)
    r = requests.post("https://api.ultralytics.com/get-export",
                      json={
                          "apiKey": api_key,
                          "modelId": model_id,
                          "format": format})
    assert (r.status_code == 200), f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    return r.json()


# temp. For checking
if __name__ == "__main__":
    start()
