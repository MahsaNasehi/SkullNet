"""Thin direct-evaluator adapter; set artifact paths through constructor arguments."""
from fracture import FracturePredictor


class Model:
    def __init__(self, **predictor_kwargs): self.fracture_predictor = FracturePredictor(**predictor_kwargs)
    def predict(self, study_dir: str) -> dict[str, float]: return {"fracture_prob": self.fracture_predictor.predict(study_dir)}

