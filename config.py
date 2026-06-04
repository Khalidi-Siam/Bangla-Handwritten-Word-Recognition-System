from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    dataset_path: Path = Path("BanglaLekha-Isolated/Images")
    labels_json_path: Path = Path("labels.json")


class TrainConfig(BaseModel):
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-4
    sample_per_class: Optional[int] = 100
    model_save_path: Path = Path("models/model.keras")

class MlflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "BanglaHandwrittenWordRecognition"


class PredictConfig(BaseModel):
    model_path: Path = Path("models/model.keras")
    labels_json_path: Path = Path("models/labels.json")
    image_size: int = 224


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