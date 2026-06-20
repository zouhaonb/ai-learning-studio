"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageSquare, LayoutDashboard, BookOpen, GitBranch, BarChart3, Sparkles } from "lucide-react";

const navItems = [
  { href: "/", label: "AI对话", icon: MessageSquare },
  { href: "/learning-path", label: "学习路径", icon: GitBranch },
  { href: "/dashboard", label: "学习仪表盘", icon: LayoutDashboard },
  { href: "/resources", label: "资源中心", icon: BookOpen },
  { href: "/knowledge-graph", label: "知识图谱", icon: GitBranch },
  { href: "/assessment", label: "学习评估", icon: BarChart3 },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">AI智学</h1>
            <p className="text-xs text-gray-500">个性化学习系统</p>
          </div>
        </div>
      </div>
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link key={item.href} href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${isActive ? "bg-blue-50 text-blue-700 shadow-sm" : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"}`}>
              <item.icon className={`w-5 h-5 ${isActive ? "text-blue-600" : "text-gray-400"}`} />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-gray-100">
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-4">
          <p className="text-xs text-gray-600 font-medium">第十五届中国软件杯</p>
          <p className="text-xs text-gray-400 mt-1">科大讯飞 · A组</p>
        </div>
      </div>
    </aside>
  );
}
