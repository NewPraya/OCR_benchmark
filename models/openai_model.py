import os
import time
import random
import base64
from openai import OpenAI
from openai import (
    OpenAIError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from dotenv import load_dotenv
from models.base import BaseOCRModel

# Load API key from .env
load_dotenv()


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _describe_openai_error(e: Exception) -> str:
    # Prefer actionable, non-noisy info (status code + request id when available)
    if isinstance(e, APIStatusError):
        rid = None
        try:
            rid = e.response.headers.get("x-request-id")
        except Exception:
            rid = None
        status = getattr(e, "status_code", None)
        base = f"{type(e).__name__} (status={status})"
        if rid:
            base += f" request_id={rid}"
        # message is often already user-friendly; keep short
        msg = str(e)
        if msg:
            base += f": {msg}"
        return base

    if isinstance(e, APITimeoutError):
        return f"{type(e).__name__}: request timed out"

    if isinstance(e, APIConnectionError):
        # openai-python wraps underlying httpx exceptions; surface the root cause if present
        cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
        if cause:
            return f"{type(e).__name__}: connection error (cause={type(cause).__name__}: {cause})"
        return f"{type(e).__name__}: connection error"

    if isinstance(e, AuthenticationError):
        return f"{type(e).__name__}: authentication failed (check OPENAI_API_KEY)"

    if isinstance(e, PermissionDeniedError):
        return f"{type(e).__name__}: permission denied (model access / org policy)"

    if isinstance(e, OpenAIError):
        return f"{type(e).__name__}: {e}"

    return f"{type(e).__name__}: {e}"


class OpenAIOCRModel(BaseOCRModel):
    def __init__(self, model_id='gpt-4o'):
        super().__init__(model_name=model_id)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        # Configurable networking knobs (OCR + images can take longer than default timeouts)
        # - SDK-level retries: handled by openai-python for some transient errors
        # - Outer retries: we handle explicit timeouts/5xx with exponential backoff
        self.timeout_seconds = _env_float("OPENAI_TIMEOUT_SECONDS", 120.0)
        self.sdk_max_retries = _env_int("OPENAI_MAX_RETRIES", 2)
        self.max_attempts = _env_int("OPENAI_OCR_MAX_ATTEMPTS", 3)
        self.backoff_base = _env_float("OPENAI_RETRY_BACKOFF_SECONDS", 1.0)
        self.backoff_max = _env_float("OPENAI_RETRY_BACKOFF_MAX_SECONDS", 20.0)
        self.backoff_jitter = _env_float("OPENAI_RETRY_BACKOFF_JITTER_SECONDS", 0.3)
        self.verbose_retries = _env_bool("OPENAI_VERBOSE_RETRIES", False)
        self.fallback_to_chat = _env_bool("OPENAI_FALLBACK_TO_CHAT", True)

        base_url = os.getenv("OPENAI_BASE_URL")  # optional (proxy / gateway)
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
            timeout=self.timeout_seconds,
            max_retries=self.sdk_max_retries,
        )

    def _sleep_backoff(self, attempt_idx: int) -> None:
        # attempt_idx: 1-based index of the *failed* attempt
        delay = min(self.backoff_max, self.backoff_base * (2 ** max(0, attempt_idx - 1)))
        delay += random.uniform(0.0, self.backoff_jitter)
        time.sleep(delay)

    def _with_retries(self, fn, api_label: str):
        last_exc = None
        for attempt in range(1, max(1, self.max_attempts) + 1):
            try:
                return fn()
            except RateLimitError as e:
                last_exc = e
                if attempt >= self.max_attempts:
                    raise
                if self.verbose_retries:
                    print(f"  ⏳ OpenAI {api_label} rate-limited; retrying (attempt {attempt}/{self.max_attempts})")
                self._sleep_backoff(attempt)
            except (APITimeoutError, APIConnectionError) as e:
                last_exc = e
                if attempt >= self.max_attempts:
                    raise
                if self.verbose_retries:
                    print(f"  ⏳ OpenAI {api_label} timed out/connection issue; retrying (attempt {attempt}/{self.max_attempts})")
                self._sleep_backoff(attempt)
            except APIStatusError as e:
                last_exc = e
                status = getattr(e, "status_code", None)
                retryable = status is not None and int(status) >= 500
                if (not retryable) or attempt >= self.max_attempts:
                    raise
                if self.verbose_retries:
                    print(f"  ⏳ OpenAI {api_label} server error {status}; retrying (attempt {attempt}/{self.max_attempts})")
                self._sleep_backoff(attempt)
            except Exception as e:
                last_exc = e
                raise
        if last_exc:
            raise last_exc

    def predict(self, image_path: str, prompt: str) -> str:
        # Read and encode image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Determine mime type
        mime_type = 'image/jpeg'
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'

        def _call_responses() -> str:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{base64_image}",
                            },
                        ],
                    }
                ],
            )
            return getattr(response, "output_text", "") or ""

        def _call_chat() -> str:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                            },
                        ],
                    }
                ],
            )
            if response.choices:
                return response.choices[0].message.content or ""
            return ""

        # Prefer Responses API for newer/experimental models
        if self.model_name.startswith(("gpt-5", "gpt-4.1", "o")):
            try:
                text = self._with_retries(_call_responses, "responses")
                if text:
                    return text
            except (BadRequestError, NotFoundError) as e:
                # Endpoint/model mismatch: optionally fall back to chat.completions
                if not self.fallback_to_chat:
                    print(f"  ❌ OpenAI responses failed: {_describe_openai_error(e)}")
                    return ""
                # continue to chat path
                if self.verbose_retries:
                    print(f"  ↪️ Falling back to chat.completions after responses error: {_describe_openai_error(e)}")
            except Exception as e:
                # For timeouts/5xx etc, we still optionally fall back (but keep error visibility)
                if not self.fallback_to_chat:
                    print(f"  ❌ OpenAI responses failed: {_describe_openai_error(e)}")
                    return ""
                if self.verbose_retries:
                    print(f"  ↪️ Falling back to chat.completions after responses error: {_describe_openai_error(e)}")

        # Chat Completions API
        try:
            return self._with_retries(_call_chat, "chat.completions")
        except Exception as e:
            extra = ""
            if self.verbose_retries:
                extra = f" (base_url={self.base_url}, timeout={self.timeout_seconds}s)"
            print(f"  ❌ OpenAI chat.completions failed: {_describe_openai_error(e)}{extra}")
            return ""
