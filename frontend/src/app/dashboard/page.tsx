"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { LayoutDashboard, TrendingUp, BookOpen, Target, Brain, Zap, Loader2, ArrowRight } from "lucide-react";
import ResourceList from "@/components/ResourceList";

interface PathNode { id: string; name: string; level: number; difficulty: number; status: string; score: number; attempts: number; }
interface PathStats { total: number; mastered: number; in_progress: number; not_started: number; avg_score: number; progress_percent: number; }
interface Resource { id: string; title: string; type: string; topic: string; difficulty: number; description: string; agent: string; duration: string; topic_ids?: string[]; }

function RadarChart({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data);
  if (entries.length === 0) return <p className="text-gray-400 text-sm text-center py-8">暂无测试数据，请先完成学习测试</p>;
  const cx = 130, cy = 130, r = 90, n = entries.length;
  const angles = entries.map((_, i) => (Math.PI * 2 * i) / n - Math.PI / 2);
  const points = angles.map((a, i) => ({ x: cx + r * (entries[i][1] / 100) * Math.cos(a), y: cy + r * (entries[i][1] / 100) * Math.sin(a) }));
  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ") + " Z";
  return (
    <svg width="260" height="260" viewBox="0 0 260 260">
      {[0.25, 0.5, 0.75, 1].map((s) => (
        <polygon key={s} points={angles.map((a) => `${cx + r * s * Math.cos(a)},${cy + r * s * Math.sin(a)}`).join(" ")} fill="none" stroke="#E5E7EB" />
      ))}
      <path d={pathD} fill="rgba(59,130,246,0.2)" stroke="#3B82F6" strokeWidth="2" />
      {points.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r="4" fill="#3B82F6" />)}
      {entries.map(([name], i) => {
        const a = angles[i]; return <text key={i} x={cx + (r + 22) * Math.cos(a)} y={cy + (r + 22) * Math.sin(a)} textAnchor="middle" fontSize="10" fill="#6B7280">{name}</text>;
      })}
    </svg>
  );
}

export default function DashboardPage() {
  const [nodes, setNodes] = useState<PathNode[]>([]);
  const [stats, setStats] = useState<PathStats | null>(null);
  const [activities, setActivities] = useState<Array<{ topic: string; type: string; time: string; status: string }>>([]);
  const [interests, setInterests] = useState<string[]>([]);
  const [recommendations, setRecommendations] = useState<Resource[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    Promise.all([
      fetch("http://localhost:8000/api/progress/path/default_user").then((r) => { if (!r.ok) throw new Error("progress"); return r.json(); }),
      fetch("http://localhost:8000/api/profile/default_user").then((r) => { if (!r.ok) throw new Error("profile"); return r.json(); }),
      fetch("http://localhost:8000/api/resources/recommendations/default_user").then((r) => { if (!r.ok) throw new Error("recommendations"); return r.json(); }),
    ]).then(([pathData, profileData, recData]) => {
      setNodes(pathData.nodes || []);
      setStats(pathData.stats || null);
      setActivities(profileData.recent_activities || []);
      setInterests(profileData.interests || []);
      setRecommendations(recData.recommendations || []);
    }).catch((e) => { console.error("Dashboard load failed:", e); })
    .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin text-blue-500" /></div>;

  // 从实际测试数据构建知识掌握度
  const knowledgeLevel: Record<string, number> = {};
  nodes.filter((n) => n.attempts > 0).forEach((n) => { knowledgeLevel[n.name] = n.score; });

  const testedNodes = nodes.filter((n) => n.attempts > 0);
  const weakNodes = nodes.filter((n) => n.attempts > 0 && n.score < 50);
  const strongNodes = nodes.filter((n) => n.score >= 70);
  const nextNode = nodes.find((n) => n.status === "in_progress") || nodes.find((n) => n.status === "not_started");

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
          <LayoutDashboard className="w-5 h-5 text-white" />
        </div>
        <div><h2 className="text-lg font-semibold text-gray-900">学习仪表盘</h2><p className="text-xs text-gray-500">基于你的实际学习数据</p></div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "已测试知识点", value: `${testedNodes.length}/${nodes.length}`, icon: BookOpen, color: "text-blue-500" },
          { label: "平均掌握度", value: stats && testedNodes.length > 0 ? `${stats.avg_score}%` : "暂无", icon: TrendingUp, color: "text-green-500" },
          { label: "薄弱环节", value: `${weakNodes.length}个`, icon: Target, color: "text-red-500" },
          { label: "学习进度", value: stats ? `${stats.progress_percent}%` : "0%", icon: Zap, color: "text-purple-500" },
        ].map((s) => (
          <div key={s.label} className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-2"><span className="text-xs text-gray-500">{s.label}</span><s.icon className={`w-4 h-4 ${s.color}`} /></div>
            <p className="text-2xl font-bold text-gray-900">{s.value}</p>
          </div>
        ))}
      </div>

      {testedNodes.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
          <BookOpen className="w-10 h-10 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500 mb-2">暂无学习数据</p>
          <p className="text-sm text-gray-400 mb-4">前往学习路径，选择知识点开始学习并完成测试</p>
          <button onClick={() => router.push("/learning-path")}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm">前往学习路径</button>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2"><Brain className="w-4 h-4" /> 知识掌握度雷达图</h3>
              <div className="flex justify-center"><RadarChart data={knowledgeLevel} /></div>
            </div>
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-4">最近学习活动</h3>
              <div className="space-y-3">
                {activities.length > 0 ? activities.map((r, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-gray-50">
                    <div className={`w-2 h-2 rounded-full ${r.status === "completed" ? "bg-green-500" : r.status === "in_progress" ? "bg-yellow-500" : "bg-gray-300"}`} />
                    <div className="flex-1"><p className="text-sm font-medium text-gray-800">{r.topic}</p><p className="text-xs text-gray-500">{r.type} · {r.time}</p></div>
                    <span className={`text-xs px-2 py-1 rounded-full ${r.status === "completed" ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-500"}`}>
                      {r.status === "completed" ? "已完成" : "进行中"}
                    </span>
                  </div>
                )) : <p className="text-sm text-gray-400 text-center py-4">暂无活动记录</p>}
              </div>
              {interests.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <p className="text-xs text-gray-500 mb-2">兴趣方向</p>
                  <div className="flex flex-wrap gap-2">{interests.map((t) => <span key={t} className="text-xs px-2 py-1 bg-purple-50 text-purple-700 rounded-full">{t}</span>)}</div>
                </div>
              )}
            </div>
          </div>

          {/* 资源推荐区域 */}
          {weakNodes.length > 0 && recommendations.length > 0 && (
            <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl border border-amber-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <Target className="w-4 h-4 text-amber-600" />
                  薄弱环节推荐资源
                </h3>
                <button
                  onClick={() => router.push("/resources")}
                  className="text-xs text-amber-600 hover:text-amber-800 underline"
                >
                  查看全部资源
                </button>
              </div>
              <ResourceList
                resources={recommendations}
                compact={true}
                maxItems={4}
                viewAllLink="/resources"
              />
            </div>
          )}

          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-4">各知识点掌握进度</h3>
            <div className="grid grid-cols-2 gap-4">
              {[...nodes].sort((a, b) => b.score - a.score).map((n) => (
                <div key={n.id} onClick={() => router.push(`/learn/${n.id}`)} className="cursor-pointer hover:bg-gray-50 p-2 rounded-lg transition-colors">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-700">{n.name}</span>
                    <span className="text-gray-500">{n.attempts > 0 ? `${n.score}%` : "未测试"}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className={`h-2 rounded-full ${n.attempts === 0 ? "bg-gray-300" : n.score >= 70 ? "bg-green-500" : n.score >= 50 ? "bg-blue-500" : "bg-red-500"}`} style={{ width: `${n.attempts > 0 ? n.score : 0}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
