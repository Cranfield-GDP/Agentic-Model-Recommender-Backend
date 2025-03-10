from pydantic import BaseModel
from typing import Optional, List


class UserInput(BaseModel):
    message: str
    model: Optional[str] = None
    userId: str

class Model(BaseModel):
    model: str
    modelLink: str

class SelectedModel(BaseModel):
    model: str
    user_id: str