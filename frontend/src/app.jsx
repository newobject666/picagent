import React, { useEffect, useState } from "react";
import {
  sendChatMessage,
  getSessions,
  getSessionMessages,
  getPaperCount,
  syncPapers,
  reloadRag,
  uploadPapersJson,
  uploadChatDocument
} from "./api";

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [paperCount, setPaperCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [documentFile, setDocumentFile] = useState(null);
  const [documentUploading, setDocumentUploading] = useState(false);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);

  async function refreshSessions() {
    const data = await getSessions();
    setSessions(data);
  }

  async function refreshPaperCount() {
    const data = await getPaperCount();
    setPaperCount(data.count);
  }

  useEffect(() => {
    refreshSessions();
    refreshPaperCount();
  }, []);

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setLoading(true);

    const userMessage = {
      role: "user",
      content: text
    };

    setMessages((prev) => [...prev, userMessage]);

    try {
      const documentIds = uploadedDocuments.map((document) => document.id);
      const data = await sendChatMessage(sessionId, text, documentIds);

      setSessionId(data.session_id);

      const assistantMessage = {
        role: "assistant",
        content: data.answer
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setUploadedDocuments([]);

      await refreshSessions();
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "请求失败：" + String(e)
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectSession(id) {
    setSessionId(id);
    const data = await getSessionMessages(id);
    setMessages(data);
  }

  async function handleSyncPapers() {
    const data = await syncPapers();
    alert(`同步完成，新增/覆盖 ${data.created} 条论文`);
    await refreshPaperCount();
  }

  async function handleReloadRag() {
    const data = await reloadRag();
    alert(data.detail);
  }

  async function handleUploadPapers() {
    if (!uploadFile || uploading) return;

    setUploading(true);
    try {
      const data = await uploadPapersJson(uploadFile);
      alert(`${data.detail}，同步 ${data.created} 条论文`);
      setUploadFile(null);
      await refreshPaperCount();
    } catch (e) {
      const detail = e.response?.data?.detail || String(e);
      alert("上传失败：" + detail);
    } finally {
      setUploading(false);
    }
  }

  async function handleUploadDocument() {
    if (!documentFile || documentUploading) return;

    setDocumentUploading(true);
    try {
      const data = await uploadChatDocument(documentFile, sessionId);
      setUploadedDocuments((prev) => [...prev, data.document]);
      setDocumentFile(null);
    } catch (e) {
      const detail = e.response?.data?.detail || String(e);
      alert("文档上传失败：" + detail);
    } finally {
      setDocumentUploading(false);
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>PicAgent</h2>

        <button
          className="new-btn"
          onClick={() => {
            setSessionId(null);
            setMessages([]);
          }}
        >
          新会话
        </button>

        <div className="paper-box">
          <div>论文库数量：{paperCount}</div>
          <label className="upload-field">
            <span>{uploadFile ? uploadFile.name : "选择 papers.json"}</span>
            <input
              type="file"
              accept=".json,application/json"
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
            />
          </label>
          <button onClick={handleUploadPapers} disabled={!uploadFile || uploading}>
            {uploading ? "上传中..." : "上传 papers.json"}
          </button>
          <button onClick={handleSyncPapers}>同步 papers.json</button>
          <button onClick={handleReloadRag}>重新加载 RAG</button>
        </div>

        <h3>历史会话</h3>
        <div className="paper-box">
          <div>本轮文档上下文：{uploadedDocuments.length}</div>
          <label className="upload-field">
            <span>{documentFile ? documentFile.name : "选择文档"}</span>
            <input
              type="file"
              accept=".txt,.md,.markdown,.docx,.pdf,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(e) => setDocumentFile(e.target.files?.[0] || null)}
            />
          </label>
          <button onClick={handleUploadDocument} disabled={!documentFile || documentUploading}>
            {documentUploading ? "解析中..." : "上传并解析文档"}
          </button>
          {uploadedDocuments.length > 0 && (
            <div className="document-list">
              {uploadedDocuments.map((document) => (
                <div key={document.id} className="document-item">
                  <span>{document.filename}</span>
                  <button
                    type="button"
                    onClick={() =>
                      setUploadedDocuments((prev) =>
                        prev.filter((item) => item.id !== document.id)
                      )
                    }
                  >
                    移除
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="session-list">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`session-item ${sessionId === s.id ? "active" : ""}`}
              onClick={() => handleSelectSession(s.id)}
            >
              {s.title}
            </div>
          ))}
        </div>
      </aside>

      <main className="main">
        <header className="header">
          <h1>科研助手 Agent</h1>
          <p>论文搜索 · 论文阅读 · 创新点分析 · 模型图生成</p>
        </header>

        <section className="messages">
          {messages.map((m, idx) => (
            <div key={idx} className={`message ${m.role}`}>
              <div className="role">{m.role === "user" ? "你" : "Agent"}</div>
              <pre>{m.content}</pre>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="role">Agent</div>
              <pre>正在思考...</pre>
            </div>
          )}
        </section>

        <footer className="input-bar">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入科研问题，例如：帮我分析 Mamba 在视觉模型中的创新点"
            onKeyDown={(e) => {
              if (e.key === "Enter" && e.ctrlKey) {
                handleSend();
              }
            }}
          />
          <button onClick={handleSend} disabled={loading}>
            发送
          </button>
        </footer>
      </main>
    </div>
  );
}
