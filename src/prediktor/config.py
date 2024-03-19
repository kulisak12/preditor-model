import os

import dotenv


class Config:
    dict_path: str = ""
    model_path: str = ""
    max_length: int = 30
    max_infill_length: int = 8
    top_k: int = 10
    temperature: float = 0.7
    confidence: float = 2.5
    num_beams: int = 10


dotenv.load_dotenv()
# load config from environment variables PREDIKTOR_MODEL_DIR, etc.
for field in Config.__dict__:
    if not field.startswith("__"):
        env_var_name = "PREDIKTOR_" + field.upper()
        if env_var_name in os.environ:
            field_type = type(getattr(Config, field))
            env_var_value = os.environ[env_var_name]
            setattr(Config, field, field_type(env_var_value))
