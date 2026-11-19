import json
import logging

from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.db import close_old_connections
from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.text import slugify
from django.views.decorators.http import require_POST

from .clinical_rag import config, pipeline
from .clinical_rag.generation import GenerationError
from .clinical_rag.knowledge_base import list_documents
from .clinical_rag.store import KnowledgeBaseNotReady
from .clinical_rag.sufficiency import SufficiencyError
from .forms import RegisterForm
from .models import ChatMessage

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000


def _source_url(source):
    """Link a source card to its document reading page."""
    filename = source.get("source", "")
    stem = filename[:-4] if filename.lower().endswith(".txt") else filename
    return reverse("chatapp:document", kwargs={"slug": slugify(stem)})


@login_required
def document_view(request, slug):
    """Render the knowledge base document as a styled reading page."""
    folder = config.KNOWLEDGE_BASE_PATH
    if not folder.is_dir():
        raise Http404("Knowledge base not found")

    for path in folder.glob("*.txt"):
        if path.is_file() and slugify(path.stem) == slug:
            content = path.read_text(encoding="utf-8").strip()
            return render(
                request,
                "chatapp/document.html",
                {
                    "doc": {
                        "stem": path.stem,
                        "filename": path.name,
                        "slug": slug,
                        "content": content,
                    }
                },
            )
    raise Http404("Document not found")


# ---------------------------------------------------------------------------
# Public pages and accounts
# ---------------------------------------------------------------------------

def landing_view(request):
    if request.user.is_authenticated:
        return redirect("chatapp:chat")
    return render(request, "chatapp/landing.html")


def register_view(request):
    if request.user.is_authenticated:
        return redirect("chatapp:chat")

    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("chatapp:chat")
    return render(request, "chatapp/registration/register.html", {"form": form})


@require_POST
def logout_view(request):
    logout(request)
    return redirect("chatapp:landing")


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

@login_required
def chat_view(request):
    recent = list(ChatMessage.objects.filter(owner=request.user).order_by("-created_at", "-id")[:50])
    recent.reverse()
    return render(
        request,
        "chatapp/chat.html",
        {"chat_messages": recent, "documents": list_documents()},
    )


@login_required
def knowledge_base_view(request):
    """Read-only list of the papers the chat can draw on."""
    return render(request, "chatapp/knowledge_base.html", {"documents": list_documents()})


@login_required
@require_POST
def ask_view(request):
    user_id = request.user.id
    question = request.POST.get("question", "").strip()
    message_id = request.POST.get("message_id", "").strip()   # set on retry and edit

    if message_id and not message_id.isdigit():
        return JsonResponse({"error": "Invalid message."}, status=400)

    chat = ChatMessage.objects.filter(owner_id=user_id)

    original = None
    if message_id:
        original = chat.filter(pk=message_id, role=ChatMessage.Role.USER).first()
        if not original:
            return JsonResponse({"error": "Message not found."}, status=404)
        question = question or original.content

    if not question:
        return JsonResponse({"error": "Please type a question."}, status=400)
    if len(question) > MAX_QUESTION_CHARS:
        return JsonResponse(
            {"error": f"Keep questions under {MAX_QUESTION_CHARS} characters."}, status=400
        )

    if original:
        # Cut the conversation back to this message; the answer is regenerated below.
        chat.filter(id__gte=original.id).delete()

    recent = list(chat.order_by("-created_at", "-id")[:6])
    history = [{"role": m.role, "content": m.content} for m in reversed(recent)]

    user_message = ChatMessage.objects.create(
        owner_id=user_id, role=ChatMessage.Role.USER, content=question
    )

    def event_stream():
        yield json.dumps({"type": "saved", "user_message_id": user_message.id}) + "\n"

        answer_parts, sources = [], []
        abstained = False
        errored = False
        abstain_details = None
        abstain_reason_message = None
        try:
            for event in pipeline.stream_rag_answer(question, history=history):
                kind = event["type"]
                if kind == "query":
                    logger.info("Searching for: %s", event["text"])
                elif kind == "sources":
                    for source in event["sources"]:
                        source["url"] = _source_url(source)
                    sources = event["sources"]
                elif kind == "token":
                    answer_parts.append(event["text"])
                elif kind == "abstain":
                    abstained = True
                    abstain_details = event.get("details", {})
                    # Capture the formal reason message for scope-based abstentions
                    abstain_reason_message = event.get("reason")
                elif kind == "error":
                    errored = True
                yield json.dumps(event) + "\n"
        except (KnowledgeBaseNotReady, GenerationError) as exc:
            logger.debug("Caught KnowledgeBaseNotReady/GenerationError: %s", exc)
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return
        except SufficiencyError as exc:
            # Operational sufficiency error (Model C missing, inference failure) - fail closed
            logger.debug("Caught SufficiencyError: %s", exc)
            yield json.dumps({"type": "error", "message": f"Sufficiency evaluation failed: {exc}"}) + "\n"
            return
        except Exception:
            logger.exception("Chat answer failed with unexpected exception")
            yield json.dumps(
                {"type": "error", "message": "Something went wrong while answering. Check the server terminal."}
            ) + "\n"
            return

        if errored:
            # Operational error (e.g., Model C failure): do not persist an assistant message
            pass
        elif abstained:
            # Persist abstention as an assistant message with explanation
            # Check if this is a scope-based abstention
            scope_decision = abstain_details.get("scope_decision")
            scope_error = abstain_details.get("scope_error")

            if scope_decision in ("OUT_OF_SCOPE", "NEEDS_CLARIFICATION"):
                # Scope classification abstention: use the formal message from the event reason
                abstain_reason = abstain_reason_message or "Scope detection failed."
            elif scope_error is not None:
                # Scope detection model failure (loading/inference error): persist formal error message
                abstain_reason = abstain_reason_message or scope.SCOPE_DETECTION_ERROR_MESSAGE
            else:
                # Existing sufficiency-based abstention (insufficient evidence)
                abstain_reason = (
                    "I cannot provide a reliable answer because the retrieved evidence "
                    "was insufficient to answer your question. The combined sufficiency "
                    f"score ({abstain_details.get('combined_score', 0):.4f}) is below "
                    f"the required threshold ({abstain_details.get('threshold', 0.70):.2f})."
                )
            close_old_connections()
            ChatMessage.objects.create(
                owner_id=user_id,
                role=ChatMessage.Role.ASSISTANT,
                content=abstain_reason,
                sources=sources,
            )
        else:
            # Normal answer generation
            close_old_connections()
            ChatMessage.objects.create(
                owner_id=user_id,
                role=ChatMessage.Role.ASSISTANT,
                content="".join(answer_parts),
                sources=sources,
            )
        yield json.dumps({"type": "done"}) + "\n"

    response = StreamingHttpResponse(event_stream(), content_type="application/x-ndjson")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@login_required
@require_POST
def clear_chat_view(request):
    ChatMessage.objects.filter(owner=request.user).delete()
    return redirect("chatapp:chat")
