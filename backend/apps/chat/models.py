from django.db import models


class ChatSession(models.Model):
    title = models.CharField(max_length=255, default="新会话")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    current_task = models.CharField(max_length=100, blank=True, default="")
    last_skill = models.CharField(max_length=100, blank=True, default="")

    class Meta:
        db_table = "chat_session"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.id} - {self.title}"


class ChatMessage(models.Model):
    ROLE_CHOICES = (
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    )

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chat_message"
        ordering = ["created_at"]


class LongTermMemory(models.Model):
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.SET_NULL,
        related_name="long_term_memories",
        null=True,
        blank=True,
    )
    category = models.CharField(max_length=100, blank=True, default="preference")
    content = models.TextField()
    importance = models.IntegerField(default=1)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "long_term_memory"
        ordering = ["-importance", "-updated_at"]


class UploadedDocument(models.Model):
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.SET_NULL,
        related_name="uploaded_documents",
        null=True,
        blank=True,
    )
    filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100, blank=True, default="")
    extension = models.CharField(max_length=20, blank=True, default="")
    extracted_text = models.TextField()
    char_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "uploaded_document"
        ordering = ["-created_at"]

    def __str__(self):
        return self.filename


class AgentTaskRecord(models.Model):
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="tasks",
    )
    user_message = models.TextField()
    task_type = models.CharField(max_length=100, blank=True, default="")
    triggered_skills = models.CharField(max_length=255, blank=True, default="")
    used_rag = models.BooleanField(default=False)
    used_web_search = models.BooleanField(default=False)
    status = models.CharField(max_length=50, default="success")
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "agent_task_record"
        ordering = ["-created_at"]


class ToolCallRecord(models.Model):
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="tool_calls",
    )
    tool_name = models.CharField(max_length=100)
    query = models.TextField(blank=True, default="")
    status = models.CharField(max_length=50, default="success")
    result_summary = models.TextField(blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "tool_call_record"
        ordering = ["-created_at"]
