"""POST /api/diarize/{video_id} — speaker diarization."""

import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    # Step 1: Extract audio
    video_path = settings.videos_dir / f"{title}.mp4"
    audio_path = diar_dir / f"{title}.wav"
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-y", str(audio_path)],
        check=True, capture_output=True,
    )

    # Step 2: Run diarization
    diar_segments = _alignment_service.diarize(str(audio_path))

    # Step 3: Unique speakers
    speakers = sorted(set(s["speaker"] for s in diar_segments))

    # Step 4: Cache result
    diar_path.write_text(json.dumps({"speakers": speakers, "segments": diar_segments}))

    # Step 5: Merge speaker labels into transcription (Task 3)
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
        labeled_segments = assign_speakers(transcript.get("segments", []), diar_segments)
        transcript["segments"] = labeled_segments
        transcript_path.write_text(json.dumps(transcript))

    return DiarizeResponse(video_id=video_id, speakers=speakers, segments=diar_segments)
