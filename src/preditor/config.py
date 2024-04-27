import os

import dotenv


class Config:
    model_path: str = ""
    fasttext_path: str = ""
    tagger_path: str = ""


dotenv.load_dotenv()
# load config from environment variables PREDITOR_MODEL_PATH, etc.
for field in Config.__dict__:
    if not field.startswith("__"):
        env_var_name = "PREDITOR_" + field.upper()
        if env_var_name in os.environ:
            field_type = type(getattr(Config, field))
            env_var_value = os.environ[env_var_name]
            setattr(Config, field, field_type(env_var_value))
