from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache

# src/core/config.py -> src/core -> src -> repo root
#
# Anchoring to the file's own location rather than the process cwd is what lets
# a worker, a test, and `python -m src.rag.ingest` all agree on where `data/`
# lives. Anything resolved against cwd works right up until something is run
# from a different directory, and then fails by silently finding nothing.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding='utf-8', extra="ignore"
    )

    #secrets nd models
    groq_api_key: str = Field( default="",validation_alias=AliasChoices("GROQ_API_KEY", "GROQ_API_KEYS"))
    llm_model : str = Field(default="openai/gpt-oss-120b",alias="LLM_MODEL")
    grader_model : str = Field(default="", alias= "GRADER_MODEL")

    llm_temperature : float = Field(default=0.0,alias="LLM_TEMPERATURE")
    grader_temperature : float = Field(default=0.0,alias="GRADER_TEMPERATURE")
    # embedding_model : str = Field(
    #     default="ibm-granite/granite-embedding-107m-multilingual",
    #     alias="EMBEDDING_MODEL"
    # )
    embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2",alias="EMBEDDING_MODEL")
    Streaming : bool =Field(default=False, alias="STREAMING")

    hf_token : str = Field(default="", alias="HF_TOKEN")
    hf_hub_offline : bool = Field(default=False, alias="HF_HUB_OFFLINE")

    supadata_api_key : str = Field(default="",alias="SUPADATA_API_KEY")
    
    retrive_K : int = Field(default=5,alias="RETRIVE_K")

    # Retrieval policy. These live here rather than in the node so the eval
    # runner and the graph exercise the same settings -- a knob that only
    # exists inside a node is a knob the golden set cannot measure.
    score_threshold : float | None = Field(
        default=None, alias="SCORE_THRESHOLD",
        description="Drop matches below this cosine score. None disables it, "
                    "which means retrieval can never return empty and the "
                    "graph can never legitimately refuse. Tune with "
                    "run_retrieval_eval.py --score-threshold.",
    )
    per_video_cap : int | None = Field(
        default=None, alias="PER_VIDEO_CAP",
        description="Max chunks per video in a playlist thread, so one long "
                    "video cannot monopolise context. Ignored when the thread "
                    "scopes a single video.",
    )

    download_limit : int = Field(default=200, alias="DOWNLOAD_LIMIT",description="total max number of youtube videos transcripts that can be downloaded")

    chunk_size : int = Field(default=800,alias='CHUNK_SIZE')

    # 2, not 5. This knob was previously declared here but never read -- the
    # splitter used its own hardcoded default of 2, so every vector currently
    # in Pinecone and every number in evals/results was produced with 2.
    # The knob is now live (splitters reads it), so the default has to match
    # what the index was actually built with. Changing it invalidates both the
    # index and the eval baseline, so treat it as a re-ingest decision.
    chunk_overlap_segments : int = Field(default=2,alias="CHUNK_OVERLAP_SEGMENTS")

    pinecone_api_key : str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="youtube-rag",alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1",alias="PINECONE_REGION")

    # The URL stays in .env and is deliberately never logged. The database
    # layer validates that it is present before constructing an engine.
    database_url: str = Field(default="", alias="DATABASE_URL")

    jwt_secret_key: str = Field(default="",alias="JWT_SECRET_KEY", repr=False,)

    access_token_expire_minutes: int = Field(default=30,alias="ACCESS_TOKEN_EXPIRE_MINUTES",)

    data_dir : Path = Field(default=Path("data"),alias="DATA_DIR")
    log_level : str =Field(default="INFO",alias="LOG_LEVEL")

    # Ingestion verification. Pinecone serverless upserts are eventually
    # consistent, so "add_documents returned" is not "a query will find it".
    # Flipping a video to `ok` before its vectors are queryable produces a
    # refusal that looks exactly like a retrieval-quality bug.
    index_verify_attempts : int = Field(default=10, alias="INDEX_VERIFY_ATTEMPTS")
    index_verify_delay : float = Field(default=2.0, alias="INDEX_VERIFY_DELAY")

    @field_validator("data_dir")
    @classmethod
    def _absolutise(cls, v: Path) -> Path:
        """Resolve a relative DATA_DIR against the repo root, not the cwd."""
        return v if v.is_absolute() else PROJECT_ROOT / v

    @property
    def transcripts_dir(self) -> Path:
        """One place that knows where transcript JSON lives.

        Previously spelled `Path(f"{settings.data_dir}/transcripts")` in both
        the fetcher and the splitter -- two copies of the same path expression
        that could drift apart.
        """
        return self.data_dir / "transcripts"

    @property
    def grader_model_name(self) -> str:
        """Fall back to the main model when no dedicated grader is set."""
        return self.grader_model or self.llm_model


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
