"use client";

import { useState, useRef, useEffect, Suspense, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { Send, Bot, User, Sparkles, Loader2, MessageSquare, Plus, Trash2, Clock } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  agent?: string;
  delegatedTo?: string | null;
  timestamp: string;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

const STORAGE_KEY = "ai_learning_conversations";

function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveConversations(convs: Conversation[]) {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
}

function getConvTitle(messages: Message[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser) return "新对话";
  const text = firstUser.content.slice(0, 25);
  return text.length < firstUser.content.length ? text + "..." : text;
}

const QUICK_PROMPTS = [
  "请为我规划一条人工智能学习路径",
  "什么是卷积神经网络？",
  "帮我出几道机器学习的练习题",
  "解释一下Transformer的注意力机制",
];

function ChatPageInner() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const searchParams = useSearchParams();
  const autoSentRef = useRef(false);
  const isLoadingRef = useRef(false);
  const currentIdRef = useRef<string | null>(null);
  const loadedRef = useRef(false);

  // Keep refs in sync
  useEffect(() => { isLoadingRef.current = isLoading; }, [isLoading]);
  useEffect(() => { currentIdRef.current = currentId; }, [currentId]);

  // Load conversations on mount
  useEffect(() => {
    const convs = loadConversations();
    if (convs.length > 0) {
      setConversations(convs);
      setCurrentId(convs[0].id);
      currentIdRef.current = convs[0].id;
    }
    loadedRef.current = true;
  }, []);

  // Save conversations whenever they change (but not on initial load)
  useEffect(() => {
    if (loadedRef.current) {
      saveConversations(conversations);
    }
  }, [conversations]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations, currentId]);

  const currentConv = conversations.find((c) => c.id === currentId);
  const messages = currentConv?.messages || [];

  // 新对话：只清空当前视图，不创建对话
  const handleNewChat = useCallback(() => {
    setCurrentId(null);
    currentIdRef.current = null;
    setInput("");
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => {
      const filtered = prev.filter((c) => c.id !== id);
      const nextId = filtered.length > 0 ? filtered[0].id : null;
      setCurrentId(nextId);
      currentIdRef.current = nextId;
      return filtered;
    });
  }, []);

  // 发送消息：如果currentId为null或forceNew=true，先创建对话再发送
  const sendMessage = useCallback(async (text: string, forceNew = false) => {
    if (!text.trim() || isLoadingRef.current) return;

    let convId = currentIdRef.current;
    if (!convId || forceNew) {
      const newConv: Conversation = {
        id: Date.now().toString(), title: text.trim().slice(0, 25),
        messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
      };
      setConversations((prev) => [newConv, ...prev]);
      setCurrentId(newConv.id);
      convId = newConv.id;
      currentIdRef.current = convId;
    }

    const userMsg: Message = {
      id: Date.now().toString(), role: "user", content: text.trim(), timestamp: new Date().toISOString(),
    };

    setConversations((prev) => prev.map((c) => {
      if (c.id !== convId) return c;
      const updated = { ...c, messages: [...c.messages, userMsg], updatedAt: new Date().toISOString() };
      updated.title = getConvTitle(updated.messages);
      return updated;
    }));
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/chat/send", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.trim(), user_id: "default_user" }),
      });
      const data = await res.json();
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(), role: "assistant", content: data.reply,
        agent: data.agent, delegatedTo: data.delegated_to, timestamp: new Date().toISOString(),
      };
      setConversations((prev) => prev.map((c) => {
        if (c.id !== convId) return c;
        return { ...c, messages: [...c.messages, assistantMsg], updatedAt: new Date().toISOString() };
      }));
    } catch (e) {
      console.error("Send failed:", e);
      setConversations((prev) => prev.map((c) => {
        if (c.id !== convId) return c;
        return { ...c, messages: [...c.messages, {
          id: (Date.now() + 1).toString(), role: "assistant", content: "连接服务器失败，请确保后端已启动。",
          agent: "System", timestamp: new Date().toISOString(),
        }], updatedAt: new Date().toISOString() };
      }));
    } finally {
      setIsLoading(false);
    }
  }, []);

  // 从URL参数自动发送（forceNew=true 确保新建对话）
  useEffect(() => {
    const q = searchParams.get("q");
    if (q && !autoSentRef.current) {
      autoSentRef.current = true;
      setTimeout(() => sendMessage(q, true), 100);
    }
  }, [searchParams, sendMessage]);

  const formatTime = (iso: string) => {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return "刚刚";
    if (diffMin < 60) return `${diffMin}分钟前`;
    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}小时前`;
    return `${d.getMonth() + 1}/${d.getDate()}`;
  };

  return (
    <div className="flex h-full">
      {/* History Sidebar */}
      {showHistory && (
        <div className="w-64 border-r border-gray-200 bg-gray-50 flex flex-col">
          <div className="p-3 border-b border-gray-200">
            <button onClick={handleNewChat}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-medium transition-colors">
              <Plus className="w-4 h-4" /> 新对话
            </button>
          </div>
          <div className="flex-1 overflow-auto p-2 space-y-1">
            {conversations.map((conv) => (
              <div key={conv.id}
                onClick={() => setCurrentId(conv.id)}
                className={`group flex items-center gap-2 px-3 py-2.5 rounded-lg cursor-pointer transition-all ${
                  currentId === conv.id ? "bg-white shadow-sm border border-gray-200" : "hover:bg-white"
                }`}>
                <MessageSquare className={`w-4 h-4 flex-shrink-0 ${currentId === conv.id ? "text-blue-600" : "text-gray-400"}`} />
                <div className="flex-1 min-w-0">
                  <p className={`text-xs truncate ${currentId === conv.id ? "font-medium text-gray-900" : "text-gray-700"}`}>{conv.title}</p>
                  <p className="text-xs text-gray-400 flex items-center gap-1"><Clock className="w-3 h-3" />{formatTime(conv.updatedAt)}</p>
                </div>
                <button onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded transition-all">
                  <Trash2 className="w-3.5 h-3.5 text-gray-400 hover:text-red-500" />
                </button>
              </div>
            ))}
            {conversations.length === 0 && (
              <p className="text-xs text-gray-400 text-center py-8">暂无对话历史</p>
            )}
          </div>
        </div>
      )}

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="border-b border-gray-200 bg-white px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button onClick={() => setShowHistory(!showHistory)} className="p-2 hover:bg-gray-100 rounded-lg">
              <MessageSquare className="w-5 h-5 text-gray-500" />
            </button>
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">{currentConv?.title || "AI学习助手"}</h2>
              <p className="text-xs text-gray-500">8个智能体协同为你服务 · {messages.length} 条消息</p>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-auto px-6 py-4 space-y-4">
          {messages.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">欢迎使用AI智学</h3>
              <p className="text-gray-500 mb-8 max-w-md">我是你的AI学习助手，由8个专业智能体协同工作。</p>
              <div className="grid grid-cols-2 gap-3 max-w-lg">
                {QUICK_PROMPTS.map((p) => (
                  <button key={p} onClick={() => sendMessage(p)}
                    className="text-left p-4 rounded-xl border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-all text-sm text-gray-700">{p}</button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              {msg.role === "assistant" && (
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-white" />
                </div>
              )}
              <div className="max-w-[70%]">
                {msg.role === "assistant" && msg.delegatedTo && (
                  <div className="text-xs text-gray-400 mb-1">
                    <span className="inline-block w-2 h-2 bg-green-400 rounded-full mr-1" />
                    {msg.agent} → {msg.delegatedTo}智能体
                  </div>
                )}
                <div className={`rounded-2xl px-4 py-3 ${msg.role === "user" ? "bg-blue-600 text-white" : "bg-white border border-gray-200 text-gray-800"}`}>
                  {msg.role === "assistant" ? (
                    <div className="prose prose-sm max-w-none"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                  ) : (
                    <p className="text-sm">{msg.content}</p>
                  )}
                </div>
                <p className="text-xs text-gray-400 mt-1 px-1">{formatTime(msg.timestamp)}</p>
              </div>
              {msg.role === "user" && (
                <div className="w-8 h-8 bg-gray-200 rounded-lg flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-gray-600" />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
                <div className="flex items-center gap-2 text-gray-500 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" /> 智能体正在思考...
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="border-t border-gray-200 bg-white px-6 py-4">
          <div className="flex gap-3">
            <input value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
              placeholder="输入你的学习问题..."
              className="flex-1 rounded-xl border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading} />
            <button onClick={() => sendMessage(input)} disabled={!input.trim() || isLoading}
              className="w-12 h-12 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 rounded-xl flex items-center justify-center transition-colors">
              <Send className="w-5 h-5 text-white" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin text-blue-500" /></div>}>
      <ChatPageInner />
    </Suspense>
  );
}
