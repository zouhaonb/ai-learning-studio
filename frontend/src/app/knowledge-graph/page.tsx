"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { GitBranch, Search, ExternalLink } from "lucide-react";

interface GraphNode {
  id: string; name: string; level: number; difficulty: number; group: number;
  x?: number; y?: number; vx?: number; vy?: number;
}
interface GraphLink { source: string; target: string; relation: string; }
interface GraphData { nodes: GraphNode[]; links: GraphLink[]; }

const COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"];

export default function KnowledgeGraphPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [highlighted, setHighlighted] = useState<Set<string>>(new Set());
  const router = useRouter();

  useEffect(() => {
    fetch("http://localhost:8000/api/knowledge-graph/graph")
      .then((r) => r.json()).then(setGraphData).catch(console.error);
  }, []);

  useEffect(() => {
    if (!graphData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const W = canvas.width = canvas.offsetWidth * 2;
    const H = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    const w = W / 2, h = H / 2;

    const nodes = graphData.nodes.map((n, i) => ({
      ...n,
      x: w / 2 + Math.cos(i * 2.3) * w * 0.3,
      y: h / 2 + Math.sin(i * 2.3) * h * 0.3,
      vx: 0, vy: 0,
    }));
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    const links = graphData.links.map((l) => ({
      source: nodeMap.get(l.source)!, target: nodeMap.get(l.target)!,
    })).filter((l) => l.source && l.target);

    let animId: number, t = 0;

    function tick() {
      t++;
      for (const link of links) {
        const dx = link.target.x! - link.source.x!, dy = link.target.y! - link.source.y!;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - 100) * 0.003;
        link.source.vx! += dx / d * f; link.source.vy! += dy / d * f;
        link.target.vx! -= dx / d * f; link.target.vy! -= dy / d * f;
      }
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x! - nodes[i].x!, dy = nodes[j].y! - nodes[i].y!;
          const d = Math.sqrt(dx * dx + dy * dy) || 1;
          const f = -200 / (d * d);
          nodes[i].vx! += dx / d * f; nodes[i].vy! += dy / d * f;
          nodes[j].vx! -= dx / d * f; nodes[j].vy! -= dy / d * f;
        }
        nodes[i].vx! += (w / 2 - nodes[i].x!) * 0.001;
        nodes[i].vy! += (h / 2 - nodes[i].y!) * 0.001;
        nodes[i].vx! *= 0.9; nodes[i].vy! *= 0.9;
        nodes[i].x = Math.max(30, Math.min(w - 30, nodes[i].x! + nodes[i].vx!));
        nodes[i].y = Math.max(30, Math.min(h - 30, nodes[i].y! + nodes[i].vy!));
      }

      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "#E5E7EB"; ctx.lineWidth = 1.5;
      for (const link of links) {
        ctx.beginPath(); ctx.moveTo(link.source.x!, link.source.y!);
        ctx.lineTo(link.target.x!, link.target.y!); ctx.stroke();
        const angle = Math.atan2(link.target.y! - link.source.y!, link.target.x! - link.source.x!);
        const ax = link.target.x! - Math.cos(angle) * 18, ay = link.target.y! - Math.sin(angle) * 18;
        ctx.fillStyle = "#D1D5DB"; ctx.beginPath(); ctx.moveTo(ax, ay);
        ctx.lineTo(ax - 8 * Math.cos(angle - 0.4), ay - 8 * Math.sin(angle - 0.4));
        ctx.lineTo(ax - 8 * Math.cos(angle + 0.4), ay - 8 * Math.sin(angle + 0.4));
        ctx.fill();
      }
      for (const node of nodes) {
        const isHl = highlighted.has(node.id), isSel = selectedNode?.id === node.id;
        const r = isSel ? 16 : isHl ? 14 : 10 + node.level * 1.5;
        const color = COLORS[node.level % COLORS.length];
        if (isSel || isHl) { ctx.shadowColor = color; ctx.shadowBlur = 15; }
        ctx.beginPath(); ctx.arc(node.x!, node.y!, r, 0, Math.PI * 2);
        ctx.fillStyle = isSel ? color : isHl ? color : color + "CC";
        ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); ctx.shadowBlur = 0;
        ctx.fillStyle = isSel ? "#1F2937" : "#4B5563";
        ctx.font = isSel ? "bold 12px sans-serif" : "11px sans-serif";
        ctx.textAlign = "center"; ctx.fillText(node.name, node.x!, node.y! + r + 14);
      }
      if (t < 300) animId = requestAnimationFrame(tick);
    }
    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [graphData, selectedNode, highlighted]);

  const handleSearch = () => {
    if (!graphData || !searchTerm.trim()) { setHighlighted(new Set()); return; }
    const term = searchTerm.toLowerCase();
    const matches = new Set(graphData.nodes.filter((n) => n.name.toLowerCase().includes(term)).map((n) => n.id));
    for (const link of graphData.links) {
      if (matches.has(link.source)) matches.add(link.target);
      if (matches.has(link.target)) matches.add(link.source);
    }
    setHighlighted(matches);
  };

  const filteredNodes = graphData?.nodes.filter((n) =>
    !searchTerm || n.name.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-teal-600 rounded-lg flex items-center justify-center">
              <GitBranch className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">AI课程知识图谱</h2>
              <p className="text-xs text-gray-500">{graphData?.nodes.length || 0} 个知识点 · {graphData?.links.length || 0} 条依赖关系</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="搜索知识点..."
                className="pl-10 pr-4 py-2 rounded-lg border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 w-64" />
            </div>
            <button onClick={handleSearch} className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm hover:bg-green-700">搜索</button>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 relative bg-gray-50">
          <canvas ref={canvasRef} className="w-full h-full" />
          <div className="absolute bottom-4 left-4 bg-white rounded-xl shadow-lg p-4 border border-gray-200">
            <p className="text-xs font-semibold text-gray-700 mb-2">图谱图例</p>
            {[0, 1, 2, 3, 4, 5].map((l) => (
              <div key={l} className="flex items-center gap-2 mb-1">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[l] }} />
                <span className="text-xs text-gray-600">{["概述", "基础", "核心", "进阶", "高级", "前沿"][l]}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="w-80 border-l border-gray-200 bg-white overflow-auto">
          <div className="p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">知识点列表 ({filteredNodes.length})</h3>
            <div className="space-y-2">
              {filteredNodes.map((node) => (
                <div key={node.id}>
                  <button
                    onClick={() => setSelectedNode(selectedNode?.id === node.id ? null : node)}
                    onDoubleClick={() => router.push(`/learn/${node.id}`)}
                    className={`w-full text-left p-3 rounded-lg border transition-all ${
                      selectedNode?.id === node.id ? "border-green-300 bg-green-50" : "border-gray-200 hover:bg-gray-50"
                    }`}>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: COLORS[node.level % COLORS.length] }} />
                      <span className="text-sm font-medium text-gray-800">{node.name}</span>
                    </div>
                    <div className="mt-1 text-xs text-gray-500">
                      难度: {"★".repeat(node.difficulty)}{"☆".repeat(5 - node.difficulty)} · Level {node.level}
                    </div>
                  </button>
                  {selectedNode?.id === node.id && (
                    <div className="mt-1 p-3 bg-green-50 rounded-lg border border-green-200 text-sm space-y-1.5">
                      <div><span className="text-gray-500 text-xs">前置依赖：</span>
                        <span className="text-xs text-gray-700">{graphData?.links.filter((l) => l.target === node.id).map((l) => graphData.nodes.find((n) => n.id === l.source)?.name).filter(Boolean).join(", ") || "无"}</span>
                      </div>
                      <div><span className="text-gray-500 text-xs">后续知识：</span>
                        <span className="text-xs text-gray-700">{graphData?.links.filter((l) => l.source === node.id).map((l) => graphData.nodes.find((n) => n.id === l.target)?.name).filter(Boolean).join(", ") || "无"}</span>
                      </div>
                      <button onClick={() => router.push(`/learn/${node.id}`)}
                        className="w-full mt-2 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-medium flex items-center justify-center gap-1 transition-colors">
                        <ExternalLink className="w-3.5 h-3.5" /> 前往学习
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
