from dotenv import dotenv_values
from pydantic.v1 import SecretStr

class DotenvException(Exception):
    pass

def load_api_key(key: str = "OPENAI_API_KEY", env_path: str = ".env") -> SecretStr:
    if (api_key := dotenv_values(env_path)[key]) is None:
        raise DotenvException(f"{key} not found in .env, or .env may not exist")
    return SecretStr(api_key)