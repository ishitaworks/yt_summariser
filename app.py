import glob
import os
import re
import tempfile
import time
from collections import defaultdict, deque

from anthropic import Anthropic
from flask import Flask, jsonify, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)
client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
MODEL = os.environ.get("SUMMARY_MODEL", "claude-sonnet-5-5")
MAX_CHARS = 300_000  # safety cap for very long videos
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # tiny / base / small / medium
MAX_AUDIO_MINUTES = int(os.environ.get("MAX_AUDIO_MINUTES", "60"))  # limit for no-caption videos
ENABLE_AUDIO = os.environ.get("ENABLE_AUDIO", "1") == "1"  # set to 0 on small hosts
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_HOUR", "30"))  # per visitor IP, 0 = off
_whisper = None
_hits = defaultdict(deque)


def rate_limited(ip: str) -> bool:
    if RATE_LIMIT <= 0:
        return False
    now = time.time()
    hits = _hits[ip]
    while hits and now - hits[0] > 3600:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        return True
    hits.append(now)
    return False

ID_PATTERN = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/|live/))([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str):
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    match = ID_PATTERN.search(url)
    return match.group(1) if match else None


def get_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    transcripts = api.list(video_id)
    try:
        transcript = transcripts.find_transcript(["en", "hi"])
    except Exception:
        transcript = next(iter(transcripts))  # fall back to any available language
    fetched = transcript.fetch()
    return " ".join(snippet.text for snippet in fetched)


def get_whisper():
    global _whisper
    from faster_whisper import WhisperModel

    if _whisper is None:  # loaded once; first run downloads the model
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def transcribe_audio(video_id: str) -> str:
    """Fallback for videos without captions: download the audio and transcribe it locally."""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmp:
        options = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmp, "audio.%(ext)s"),
            "quiet": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            if (info.get("duration") or 0) > MAX_AUDIO_MINUTES * 60:
                raise ValueError(
                    f"This video has no captions and is longer than {MAX_AUDIO_MINUTES} minutes, "
                    "so it is too long to transcribe."
                )
            ydl.download([url])
        audio_path = glob.glob(os.path.join(tmp, "audio.*"))[0]
        segments, _info = get_whisper().transcribe(audio_path, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments)


def summarize(text: str) -> str:
    text = text[:MAX_CHARS]
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": (
                    "Below is the transcript of a YouTube video. Write a clear summary in English "
                    "using this format:\n\n"
                    "## Overview\n(2-3 sentences)\n\n"
                    "## Key points\n(bullet list)\n\n"
                    "## Takeaways\n(1-3 bullets)\n\n"
                    "Only use what is in the transcript.\n\n"
                    f"Transcript:\n{text}"
                ),
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    if rate_limited(ip):
        return jsonify(error="Too many requests. Try again in a little while."), 429
    url = (request.get_json(silent=True) or {}).get("url", "")
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify(error="That doesn't look like a valid YouTube link."), 400
     try:
        transcript = get_transcript(video_id)
     except Exception as exc:
        print("CAPTION ERROR:", repr(exc), flush=True)
        if not ENABLE_AUDIO:
            return jsonify(error="This video has no captions, and audio transcription is turned off on this server."), 422
        try:
            transcript = transcribe_audio(video_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 422
        except Exception as exc:
            return jsonify(error=f"No captions, and audio transcription failed: {exc}"), 422
    if not transcript.strip():
        return jsonify(error="No speech could be found in this video."), 422
    try:
        summary = summarize(transcript)
    except Exception as exc:
        return jsonify(error=f"Summary failed: {exc}"), 500
    return jsonify(summary=summary, video_id=video_id)


if __name__ == "__main__":
    app.run(debug=True)
