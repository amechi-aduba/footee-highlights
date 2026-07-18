import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import (
    CORS_ORIGINS,
    UPLOAD_CLEANUP_INTERVAL_SECONDS,
    ensure_storage_directories,
)
from app.models.schemas import HealthResponse
from app.routes.videos import router as videos_router
from app.services.video_storage import purge_expired_video_data


async def _cleanup_abandoned_uploads() -> None:
    while True:
        await asyncio.sleep(UPLOAD_CLEANUP_INTERVAL_SECONDS)
        await asyncio.to_thread(purge_expired_video_data)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_storage_directories()
    await asyncio.to_thread(purge_expired_video_data)
    cleanup_task = asyncio.create_task(_cleanup_abandoned_uploads())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="Footee Vision API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos_router)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="footee-vision-backend")
