const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatResponse {
  reply: string;
  agent: string;
  delegated_to: string | null;
  resources: Array<{ type: string; title: string; content: string }>;
}

export interface GraphData {
  nodes: Array<{ id: string; name: string; level: number; difficulty: number; group: number }>;
  links: Array<{ source: string; target: string; relation: string }>;
}

export async function sendMessage(message: string, userId = "default_user"): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, user_id: userId }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getKnowledgeGraph(): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/api/knowledge-graph/graph`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getTopics() {
  const res = await fetch(`${API_BASE}/api/knowledge-graph/topics`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getLearningPath(target: string, mastered = "") {
  const res = await fetch(`${API_BASE}/api/knowledge-graph/learning-path?target=${encodeURIComponent(target)}&mastered=${encodeURIComponent(mastered)}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
