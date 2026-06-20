"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { BarChart3, TrendingUp, AlertTriangle, CheckCircle, Loader2, ArrowRight } from "lucide-react";

interface PathNode { id: string; name: string; level: number; difficulty: number; status: string; score: number; attempts: number; wrong_topics_history?: string[]; }

export default function AssessmentPage() {
  const [nodes, setNodes] = useState<PathNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const router = useRouter();

  const loadData = () => {
    setLoading(true); setError("");
    fetch("http://localhost:8000/api/progress/path/default_user")
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d) => setNodes(d.nodes || []))
      .catch((e) => { console.error("Load failed:", e); setError("加载失败"); })
      .finally(() => setLoading(false));
  };

  useEffect(() => { loadData(); }, []);

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin text-rose-500" /></div>;
  if (error) return <div className="flex flex-col items-center justify-center h-full gap-4"><p className="text-gray-500">{error}</p><button onClick={loadData} className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm">重试</button></div>;

  const tested = nodes.filter((n) => n.attempts > 0);
  const avg = tested.length > 0 ? Math.round(tested.reduce((s, n) => s + n.score, 0) / tested.length) : 0;
  const weakTopics = tested.filter((n) => n.score < 50);
  const strongTopics = tested.filter((n) => n.score >= 70);
  const suggestions = weakTopics.slice(0, 4).map((n) => ({
    pri: n.score < 30 ? "高" : "中", topic: n.name, id: n.id,
    action: n.score < 30 ? `从基础概念开始学习，完成入门测试` : `复习核心公式，完成进阶测试`,
    reason: `当前掌握度${n.score}%，低于掌握标准`,
  }));

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-rose-500 to-pink-600 rounded-lg flex items-center justify-center">
          <BarChart3 className="w-5 h-5 text-white" />
        </div>
        <div><h2 className="text-lg font-semibold text-gray-900">学习效果评估</h2><p className="text-xs text-gray-500">基于实际测试数据生成</p></div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "综合掌握度", val: tested.length > 0 ? `${avg}%` : "暂无数据", color: "text-gray-900", bar: tested.length > 0 ? avg : undefined },
          { label: "已测试", val: `${tested.length}/${nodes.length}`, color: "text-blue-600" },
          { label: "薄弱环节", val: `${weakTopics.length}个`, color: "text-red-500" },
          { label: "已掌握", val: `${strongTopics.length}个`, color: "text-green-600" },
        ].map((c) => (
          <div key={c.label} className="bg-white rounded-xl border border-gray-200 p-4">
            <p className="text-xs text-gray-500 mb-1">{c.label}</p>
            <p className={`text-3xl font-bold ${c.color}`}>{c.val}</p>
            {c.bar !== undefined && <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2"><div className="bg-blue-600 h-1.5 rounded-full" style={{ width: `${c.bar}%` }} /></div>}
          </div>
        ))}
      </div>

      {tested.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
          <BarChart3 className="w-10 h-10 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500 mb-2">暂无测试数据</p>
          <p className="text-sm text-gray-400 mb-4">完成学习测试后，这里会显示你的评估报告</p>
          <button onClick={() => router.push("/learning-path")} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm">前往学习路径</button>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-4">各知识点掌握详情</h3>
            <div className="space-y-3">
              {[...nodes].sort((a, b) => b.score - a.score).map((n) => (
                <div key={n.id} className="flex items-center gap-3">
                  <div className="w-24 text-xs text-gray-700 truncate">{n.name}</div>
                  <div className="flex-1 bg-gray-200 rounded-full h-2.5">
                    <div className={`h-2.5 rounded-full ${n.attempts === 0 ? "bg-gray-300" : n.score >= 70 ? "bg-green-500" : n.score >= 50 ? "bg-yellow-500" : "bg-red-500"}`}
                      style={{ width: `${n.attempts > 0 ? n.score : 0}%` }} />
                  </div>
                  <span className="text-xs font-medium w-14 text-right">{n.attempts > 0 ? `${n.score}%` : "未测试"}</span>
                  {n.attempts > 0 && n.score < 50 && <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />}
                  {n.score >= 70 && n.attempts > 0 && <CheckCircle className="w-3.5 h-3.5 text-green-500" />}
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-4">个性化改进建议</h3>
              {suggestions.length > 0 ? (
                <div className="space-y-3">
                  {suggestions.map((s, i) => (
                    <div key={i}
                      className={`p-3 rounded-lg border-l-4 ${s.pri === "高" ? "border-red-500 bg-red-50" : "border-amber-500 bg-amber-50"}`}>
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${s.pri === "高" ? "bg-red-200 text-red-800" : "bg-amber-200 text-amber-800"}`}>{s.pri}</span>
                        <span className="text-sm font-semibold text-gray-800">{s.topic}</span>
                      </div>
                      <p className="text-xs text-gray-700">{s.action}</p>
                      <p className="text-xs text-gray-500 mt-0.5">{s.reason}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <button onClick={() => router.push(`/learn/${s.id}`)}
                          className="text-xs text-blue-600 hover:text-blue-800 underline flex items-center gap-1">
                          开始学习 <ArrowRight className="w-3 h-3" />
                        </button>
                        <button onClick={() => router.push(`/resources?topic=${s.id}`)}
                          className="text-xs text-amber-600 hover:text-amber-800 underline flex items-center gap-1">
                          查看相关资源 <ArrowRight className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-2 p-3 bg-green-50 rounded-lg">
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  <span className="text-sm text-green-800">所有已测试知识点掌握良好！</span>
                </div>
              )}
            </div>

            <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-semibold text-blue-800">评估说明</span>
              </div>
              <p className="text-xs text-blue-700">掌握度 = 你实际测试得分。点击薄弱环节可直接进入学习。继续完成更多测试以获得更全面的评估。</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
