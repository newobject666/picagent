import axios from "axios";

export async function sendChatMessage(sessionId, message, documentIds = []) {
  const res = await axios.post("/api/chat/", {
    session_id: sessionId,
    message,
    document_ids: documentIds
  });
  return res.data;
}

export async function getSessions() {
  const res = await axios.get("/api/chat/sessions/");
  return res.data;
}

export async function getSessionMessages(sessionId) {
  const res = await axios.get(`/api/chat/sessions/${sessionId}/messages/`);
  return res.data;
}

export async function getPaperCount() {
  const res = await axios.get("/api/papers/count/");
  return res.data;
}

export async function syncPapers() {
  const res = await axios.post("/api/papers/sync/");
  return res.data;
}

export async function uploadPapersJson(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await axios.post("/api/papers/upload/", formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });
  return res.data;
}

export async function uploadChatDocument(file, sessionId) {
  const formData = new FormData();
  formData.append("file", file);
  if (sessionId) {
    formData.append("session_id", sessionId);
  }

  const res = await axios.post("/api/chat/documents/upload/", formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });
  return res.data;
}

export async function reloadRag() {
  const res = await axios.post("/api/papers/reload-rag/");
  return res.data;
}
