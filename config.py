from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    dataset_path: Path = Path("BanglaLekha-Isolated/Images")
    labels_json_path: Path = Path("labels.json")


class TrainConfig(BaseModel):
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-4
    sample_per_class: Optional[int] = 100 # Set to None for no limit, or an integer for max samples per class during training
    model_save_path: Path = Path("models/model.keras")
    save_json_path: Path = Path("models/labels.json")
    save_summary_path: Path = Path("models/train_history.csv")

    @field_validator('sample_per_class', mode='before')
    @classmethod
    def parse_sample_per_class(cls, v):
        if isinstance(v, str) and v.lower() in ('none', 'null', ''):
            return None
        return v

class MlflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    registry_uri: str = "http://localhost:5000"
    experiment_name: str = "Bengali_Word_Recognition"


class PredictConfig(BaseModel):
    model_path: Path = Path("model.keras")
    labels_json_path: Path = Path("labels.json")
    image_size: int = 64


class Settings(BaseSettings):
    seed: int = 42
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    mlflow: MlflowConfig = MlflowConfig()
    predict: PredictConfig = PredictConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )


settings = Settings()
