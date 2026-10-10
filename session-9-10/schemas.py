from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class Question(BaseModel):
    text: str = Field(min_length=3, max_length=1000)

    @model_validator(mode="after")
    def reject_blank(self):
        self.text = self.text.strip()
        if len(self.text) < 3:
            raise ValueError("Ask a question containing at least three characters")
        return self


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(min_length=1)
    citations: list[str]

    @model_validator(mode="after")
    def citation_contract(self):
        if self.status == "answered" and not self.citations:
            raise ValueError("An answered result needs at least one passage ID")
        if self.status == "insufficient_evidence" and self.citations:
            raise ValueError("An insufficient-evidence result must have no citations")
        return self


class VectorRecord(BaseModel):
    id: str
    text: str
    metadata: dict
    vector: list[FiniteFloat] = Field(min_length=1)


class IndexSnapshot(BaseModel):
    schema_version: Literal[1] = 1
    fingerprint: str
    mode: Literal["live", "offline"]
    embedding_model: str
    records: dict[str, VectorRecord]

    @model_validator(mode="after")
    def validate_vectors(self):
        dimensions = {len(record.vector) for record in self.records.values()}
        if len(dimensions) != 1:
            raise ValueError("The index must contain vectors of one consistent dimension")
        for key, record in self.records.items():
            if key != record.id or record.metadata.get("chunk_id") != key:
                raise ValueError("Saved passage IDs must match their metadata")
        return self
