"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { BookOpen, Search, Filter, FileText, HelpCircle, Code, Video, Lightbulb, Loader2, X } from "lucide-react";

const TYPES = [
  { id: "all", label: "全部", icon: BookOpen },
  { id: "document", label: "课程文档", icon: FileText },
  { id: "quiz", label: "练习题", icon: HelpCircle },
  { id: "code", label: "代码案例", icon: Code },
  { id: "reading", label: "拓展阅读", icon: Lightbulb },
  { id: "multimedia", label: "多媒体", icon: Video },
];

interface Resource {
  id: string; title: string; type: string; topic: string; difficulty: number;
  description: string; agent: string; duration: string;
}

export default function ResourcesPage() {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [resources, setResources] = useState<Resource[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Resource | null>(null);
  const router = useRouter();

  useEffect(() => {
    fetch("http://localhost:8000/api/resources/list")
      .then((r) => r.json()).then((d) => setResources(d.resources || []))
      .catch(console.error).finally(() => setLoading(false));
  }, []);

  const filtered = resources.filter((r) => {
    if (filter !== "all" && r.type !== filter) return false;
    if (search && !r.title.includes(search) && !r.topic.includes(search)) return false;
    return true;
  });

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin text-amber-500" /></div>;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-gradient-to-br from-amber-500 to-orange-600 rounded-lg flex items-center justify-center">
          <BookOpen className="w-5 h-5 text-white" />
        </div>
        <div><h2 className="text-lg font-semibold text-gray-900">资源中心</h2><p className="text-xs text-gray-500">智能体为你生成的个性化学习资源 · 共 {resources.length} 份</p></div>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="搜索资源..."
            className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500" />
        </div>
        <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
          {TYPES.map((t) => (
            <button key={t.id} onClick={() => setFilter(t.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                filter === t.id ? "bg-white text-amber-700 shadow-sm" : "text-gray-500 hover:text-gray-700"
              }`}><t.icon className="w-3.5 h-3.5" />{t.label}</button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {filtered.map((r) => {
          const Icon = TYPES.find((t) => t.id === r.type)?.icon || BookOpen;
          return (
            <div key={r.id} onClick={() => setSelected(r)}
              className="bg-white rounded-xl border border-gray-200 p-5 hover:border-amber-300 hover:shadow-md transition-all cursor-pointer group">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-amber-50 flex items-center justify-center"><Icon className="w-4 h-4 text-amber-600" /></div>
                  <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full">{r.topic}</span>
                  <span className="text-xs text-gray-400">{"★".repeat(r.difficulty)}</span>
                </div>
                <span className="text-xs text-gray-400">{r.duration}</span>
              </div>
              <h3 className="text-sm font-semibold text-gray-900 mb-2 group-hover:text-amber-700">{r.title}</h3>
              <p className="text-xs text-gray-500 leading-relaxed mb-3">{r.description}</p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">由 {r.agent} 生成</span>
                <span className="text-xs text-amber-600 opacity-0 group-hover:opacity-100 transition-opacity">查看详情 →</span>
              </div>
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-gray-400"><Filter className="w-8 h-8 mx-auto mb-2" /><p className="text-sm">没有匹配的资源</p></div>
      )}

      {/* Detail Modal */}
      {selected && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" onClick={() => setSelected(null)}>
          <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full mx-4 p-6" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">{selected.title}</h3>
              <button onClick={() => setSelected(null)} className="p-1 hover:bg-gray-100 rounded-lg"><X className="w-5 h-5 text-gray-500" /></button>
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex gap-2">
                <span className="px-2 py-1 bg-amber-50 text-amber-700 rounded-full text-xs">{selected.type}</span>
                <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-full text-xs">{selected.topic}</span>
                <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-full text-xs">难度{"★".repeat(selected.difficulty)}</span>
              </div>
              <p className="text-gray-700">{selected.description}</p>
              <div className="pt-3 border-t border-gray-100">
                <p className="text-xs text-gray-500">生成智能体: <span className="text-gray-700 font-medium">{selected.agent}</span></p>
                <p className="text-xs text-gray-500">预计时长: <span className="text-gray-700">{selected.duration}</span></p>
              </div>
              <button onClick={() => {
                const msg = `请帮我学习"${selected.title}"这个知识点，给我详细的讲解`;
                router.push(`/?q=${encodeURIComponent(msg)}`);
              }} className="w-full mt-4 py-2.5 bg-amber-600 hover:bg-amber-700 text-white rounded-xl text-sm font-medium transition-colors">
                开始学习
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
