"""Pydantic μοντέλα για τα μηνύματα που δέχεται το API (έλεγχος εισόδου)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from services.config import FEATURE_SIZE


class ClientControl(BaseModel):
    """Μηνύματα ελέγχου (text frames) από τον client μέσω WebSocket."""

    type: Literal["config", "reset", "ping"]
    threshold: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    min_votes: Optional[int] = Field(default=None, ge=1, le=20)


class SequenceRequest(BaseModel):
    """Για το POST /api/predict/sequence: έτοιμα landmarks (T x 126), χωρίς κάμερα."""

    language: Literal["gsl", "asl", "gsl_sentences"]
    sequence: list[list[float]] = Field(min_length=1, max_length=600)

    @field_validator("sequence")
    @classmethod
    def _check_width(cls, seq: list[list[float]]) -> list[list[float]]:
        for i, row in enumerate(seq):
            if len(row) != FEATURE_SIZE:
                raise ValueError(f"Η γραμμή {i} έχει {len(row)} τιμές, αναμένονται {FEATURE_SIZE}")
        return seq


class TopK(BaseModel):
    label: str
    confidence: float


class SequencePrediction(BaseModel):
    language: str
    label: str
    confidence: float
    top_k: list[TopK]
