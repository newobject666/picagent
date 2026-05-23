from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.db import transaction
from django.db.models import Count, Q

from backend.apps.agents.agent_manage import agent_manager
from apps.chat.models import (
    ChatSession,
    ChatMessage,
    AgentTaskRecord,
    LongTermMemory,
    UploadedDocument,
)
from apps.chat.document_parser import build_document_context, parse_uploaded_document
from apps.chat.serializers import (
    ChatSessionSerializer,
    ChatMessageSerializer,
    ChatRequestSerializer,
    LongTermMemorySerializer,
    UploadedDocumentSerializer,
)
import logging

logger = logging.getLogger("picagent")


def load_active_long_term_memories(limit=20):
    return list(
        LongTermMemory.objects
        .filter(is_active=True)
        .order_by("-importance", "-updated_at")
        .values_list("content", flat=True)[:limit]
    )


def save_extracted_long_term_memories(session, memories):
    saved = 0

    for memory in memories:
        content = memory.strip()
        if not content:
            continue

        if LongTermMemory.objects.filter(
            is_active=True,
            content=content,
        ).exists():
            continue

        LongTermMemory.objects.create(
            session=session,
            category="auto",
            content=content,
            importance=1,
        )
        saved += 1

    return saved


def load_uploaded_documents_for_chat(session, document_ids):
    if not document_ids:
        return []

    documents = list(
        UploadedDocument.objects
        .filter(id__in=document_ids)
        .filter(Q(session=session) | Q(session__isnull=True))
        .order_by("created_at")
    )

    if documents:
        UploadedDocument.objects.filter(
            id__in=[document.id for document in documents],
            session__isnull=True,
        ).update(session=session)

    return documents


def build_agent_document_payload(documents):
    return [
        {
            "id": document.id,
            "filename": document.filename,
            "text": document.extracted_text,
        }
        for document in documents[:5]
    ]


def has_complete_chat_turn(user_message, assistant_answer):
    return bool(
        user_message
        and user_message.strip()
        and assistant_answer
        and assistant_answer.strip()
    )


def completed_session_queryset():
    return (
        ChatSession.objects
        .annotate(
            user_message_count=Count(
                "messages",
                filter=Q(messages__role="user"),
            ),
            assistant_message_count=Count(
                "messages",
                filter=Q(messages__role="assistant"),
            ),
        )
        .filter(user_message_count__gt=0, assistant_message_count__gt=0)
    )


@api_view(["POST"])
def chat_view(request):
    logger.info("收到 /api/chat/ 请求")

    serializer = ChatRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    message = serializer.validated_data["message"].strip()
    session_id = serializer.validated_data.get("session_id")
    document_ids = serializer.validated_data.get("document_ids", [])

    logger.info(f"用户输入: {message}")
    logger.info(f"session_id: {session_id}")

    if not message:
        return Response(
            {"detail": "message 不能为空"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    created_session = False

    if session_id:
        session = ChatSession.objects.filter(id=session_id).first()
        if not session:
            session = ChatSession.objects.create(title=message[:30])
            created_session = True
            logger.info(f"session 不存在，创建新 session: {session.id}")
    else:
        session = ChatSession.objects.create(title=message[:30])
        created_session = True
        logger.info(f"创建新 session: {session.id}")

    agent = agent_manager.get_agent(session.id)
    agent.set_long_term_memories(load_active_long_term_memories())
    documents = load_uploaded_documents_for_chat(session, document_ids)
    agent.set_runtime_documents(build_agent_document_payload(documents))
    logger.info("ResearchAgent 获取成功，已注入长期记忆，开始调用 Agent")

    try:
        answer = agent.ask(message)

        logger.info("ResearchAgent 调用完成")
        logger.info(f"Agent 回答长度: {len(answer)}")

        if not has_complete_chat_turn(message, answer):
            raise ValueError("Agent returned an empty answer; incomplete session was not saved")

        with transaction.atomic():
            ChatMessage.objects.create(
                session=session,
                role="user",
                content=message,
            )
            ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=answer,
            )
            AgentTaskRecord.objects.create(
                session=session,
                user_message=message,
                status="success",
            )

        logger.info("complete chat turn 已保存到 MySQL")

        try:
            extracted_memories = agent.extract_long_term_memories(message, answer)
            saved_memory_count = save_extracted_long_term_memories(
                session=session,
                memories=extracted_memories,
            )
            logger.info(f"长期记忆提取完成，新增 {saved_memory_count} 条")
        except Exception:
            logger.exception("长期记忆提取失败，已忽略")

        return Response(
            {
                "session_id": session.id,
                "answer": answer,
                "documents": UploadedDocumentSerializer(documents, many=True).data,
            }
        )

    except Exception as exc:
        logger.exception("Agent 调用失败")

        agent_manager.remove_agent(session.id)

        response_session_id = session.id
        if created_session and not session.messages.exists():
            response_session_id = None
            session.delete()
            logger.info("incomplete new session removed")
        else:
            try:
                AgentTaskRecord.objects.create(
                    session=session,
                    user_message=message,
                    status="failed",
                    error_message=str(exc),
                )
            except Exception:
                logger.exception("failed task record 保存失败，已忽略")

        return Response(
            {
                "session_id": response_session_id,
                "detail": str(exc),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        agent.clear_runtime_documents()

@api_view(["GET"])
def session_list_view(request):
    sessions = completed_session_queryset()
    serializer = ChatSessionSerializer(sessions, many=True)
    return Response(serializer.data)


@api_view(["GET"])
def session_messages_view(request, session_id: int):
    session = ChatSession.objects.filter(id=session_id).first()

    if not session:
        return Response(
            {"detail": "session 不存在"},
            status=status.HTTP_404_NOT_FOUND,
        )

    messages = session.messages.all()
    serializer = ChatMessageSerializer(messages, many=True)
    return Response(serializer.data)


@api_view(["POST"])
def clear_session_view(request, session_id: int):
    session = ChatSession.objects.filter(id=session_id).first()

    if not session:
        return Response(
            {"detail": "session 不存在"},
            status=status.HTTP_404_NOT_FOUND,
        )

    session.messages.all().delete()
    agent_manager.clear_agent(session.id)

    return Response({"detail": "会话已清空"})


@api_view(["GET"])
def memory_list_view(request):
    memories = LongTermMemory.objects.filter(is_active=True)
    serializer = LongTermMemorySerializer(memories, many=True)
    return Response(serializer.data)


@api_view(["DELETE"])
def memory_delete_view(request, memory_id: int):
    memory = LongTermMemory.objects.filter(id=memory_id).first()

    if not memory:
        return Response(
            {"detail": "记忆不存在"},
            status=status.HTTP_404_NOT_FOUND,
        )

    memory.is_active = False
    memory.save(update_fields=["is_active", "updated_at"])

    return Response({"detail": "记忆已删除"})
@api_view(["POST"])
def upload_document_view(request):
    uploaded_file = request.FILES.get("file")
    session_id = request.data.get("session_id")

    if uploaded_file is None:
        return Response(
            {"detail": "请选择要上传的文档"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    session = None
    if session_id:
        session = ChatSession.objects.filter(id=session_id).first()

    try:
        parsed = parse_uploaded_document(uploaded_file, uploaded_file.name)
    except ValueError as exc:
        return Response(
            {"detail": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )

    document = UploadedDocument.objects.create(
        session=session,
        filename=parsed.filename[:255],
        content_type=getattr(uploaded_file, "content_type", "") or "",
        extension=parsed.extension,
        extracted_text=parsed.text,
        char_count=parsed.char_count,
    )
    preview = build_document_context([document], max_total_chars=1200)

    return Response(
        {
            "detail": "文档上传并解析完成",
            "document": UploadedDocumentSerializer(document).data,
            "preview": preview,
        }
    )
