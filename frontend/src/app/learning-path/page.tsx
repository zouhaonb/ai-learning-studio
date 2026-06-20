"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { GitBranch, BookOpen, CheckCircle, Clock, ArrowRight, Loader2, Star, Trophy } from "lucide-react";
import ResourceBadges from "@/components/ResourceBadges";

interface PathNode {
  id: string; name: string; level: number; difficulty: number;
  status: "not_started" | "in_progress" | "mastered";
  score: number; attempts: number; prerequisites: string[];
  description: string; keywords: string[];
  resource_count?: number;
  resource_types?: string[];
}
interface PathStats { total: number; mastered: number; in_progress: number; not_started: number; avg_score: number; progress_percent: number; }

const CFG = {
  mastered: { color: "bg-green-500", border: "border-green-300", bg: "bg-green-50", icon: CheckCircle, label: "已掌握" },
  in_progress: { color: "bg-blue-500", border: "border-blue-300", bg: "bg-blue-50", icon: Clock, label: "学习中" },
  not_started: { color: "bg-gray-300", border: "border-gray-200", bg: "bg-gray-50", icon: BookOpen, label: "未开始" },
};

export default function LearningPathPage() {
  const [nodes, setNodes] = useState<PathNode[]>([]);
  const [stats, setStats] = useState<PathStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const router = useRouter();

  const loadPath = () => {
    setLoading(true); setError("");
    fetch("http://localhost:8000/api/progress/path/default_user")
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d) => { setNodes(d.nodes); setStats(d.stats); })
      .catch((e) => { console.error("Load failed:", e); setError("加载失败，请确保后端已启动"); setNodes([]); })
      .finally(() => setLoading(false));
  };

  useEffect(() => { loadPath(); }, []);

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin text-blue-500" /></div>;
  if (error) return <div className="flex flex-col items-center justify-center h-full gap-4"><p className="text-gray-500">{error}</p><button onClick={loadPath} className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm">重试</button></div>;

  const grouped = [0, 1, 2, 3, 4, 5].map((l) => nodes.filter((n) => n.level === l)).filter((g) => g.length > 0);
  const levelNames = ["基础概述", "核心领域", "基础算法", "进阶模型", "高级技术", "前沿方向"];
  const nextOne = nodes.find((n) => n.status === "in_progress") || nodes.find((n) => n.status === "not_started");

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center">
          <GitBranch className="w-5 h-5 text-white" />
        </div>
        <div><h2 className="text-lg font-semibold text-gray-900">学习路径</h2><p className="text-xs text-gray-500">AI课程完整路径 · 共 {nodes.length} 个知识点</p></div>
      </div>

      {stats && (
        <div className="grid grid-cols-5 gap-4">
          {[
            { val: stats.total, label: "总知识点", color: "text-gray-900" },
            { val: stats.mastered, label: "已掌握", color: "text-green-600" },
            { val: stats.in_progress, label: "学习中", color: "text-blue-600" },
            { val: stats.not_started, label: "未开始", color: "text-gray-400" },
            { val: `${stats.progress_percent}%`, label: "完成度", color: "text-purple-600" },
          ].map((s) => (
            <div key={s.label} className="bg-white rounded-xl border border-gray-200 p-4 text-center">
              <p className={`text-2xl font-bold ${s.color}`}>{s.val}</p><p className="text-xs text-gray-500">{s.label}</p>
            </div>
          ))}
        </div>
      )}

      {nextOne && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200 p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ArrowRight className="w-5 h-5 text-blue-600" />
            <div><p className="text-sm font-semibold text-gray-800">推荐：{nextOne.name}</p><p className="text-xs text-gray-500">{nextOne.description}</p></div>
          </div>
          <button onClick={() => router.push(`/learn/${nextOne.id}`)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium">开始学习</button>
        </div>
      )}

      <div className="space-y-8">
        {grouped.map((group, gi) => (
          <div key={gi}>
            <h3 className="text-sm font-semibold text-gray-500 mb-3 flex items-center gap-2">
              <span className="w-6 h-6 bg-gray-200 rounded-full flex items-center justify-center text-xs">{gi + 1}</span>
              {levelNames[gi] || `Level ${gi}`}
            </h3>
            <div className="grid grid-cols-3 gap-4 ml-8">
              {group.map((node) => {
                const c = CFG[node.status]; const Icon = c.icon;
                return (
                  <div key={node.id} onClick={() => router.push(`/learn/${node.id}`)}
                    className={`p-4 rounded-xl border-2 ${c.border} ${c.bg} cursor-pointer hover:shadow-md transition-all group`}>
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className={`w-8 h-8 rounded-lg ${c.color} flex items-center justify-center`}>
                          <Icon className="w-4 h-4 text-white" />
                        </div>
                        <div><p className="text-sm font-semibold text-gray-900 group-hover:text-blue-600">{node.name}</p><p className="text-xs text-gray-500">{c.label}</p></div>
                      </div>
                      {node.attempts > 0 && <div className="flex items-center gap-1"><Star className="w-4 h-4 text-yellow-500" /><span className="text-sm font-bold">{node.score}%</span></div>}
                    </div>
                    <p className="text-xs text-gray-600 mb-2 line-clamp-2">{node.description}</p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">难度 {"★".repeat(node.difficulty)}</span>
                      <div className="flex items-center gap-2">
                        {node.resource_count && node.resource_count > 0 && (
                          <ResourceBadges
                            count={node.resource_count}
                            types={node.resource_types || []}
                            onClick={(e) => {
                              e.stopPropagation();
                              router.push(`/resources?topic=${node.id}`);
                            }}
                          />
                        )}
                        {node.status === "mastered" && <Trophy className="w-4 h-4 text-yellow-500" />}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            {gi < grouped.length - 1 && <div className="ml-10 w-0.5 h-6 bg-gray-200 mt-2" />}
          </div>
        ))}
      </div>
    </div>
  );
}
