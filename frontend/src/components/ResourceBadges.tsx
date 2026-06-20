"use client";

import { FileText, HelpCircle, Code, Video, Lightbulb } from "lucide-react";

const TYPE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  document: FileText,
  quiz: HelpCircle,
  code: Code,
  reading: Lightbulb,
  multimedia: Video,
};

const TYPE_LABELS: Record<string, string> = {
  document: "文档",
  quiz: "测验",
  code: "代码",
  reading: "阅读",
  multimedia: "视频",
};

interface ResourceBadgesProps {
  count: number;
  types: string[];
  onClick?: (e: React.MouseEvent) => void;
}

export default function ResourceBadges({ count, types, onClick }: ResourceBadgesProps) {
  if (count === 0) return null;

  return (
    <div
      className={`flex items-center gap-1.5 text-xs text-gray-500 ${onClick ? "cursor-pointer hover:text-blue-600" : ""}`}
      onClick={onClick}
      title={`${count} 份资源: ${types.map(t => TYPE_LABELS[t] || t).join(", ")}`}
    >
      <span className="font-medium">📚 {count}份</span>
      <div className="flex items-center gap-0.5">
        {types.slice(0, 3).map((type) => {
          const Icon = TYPE_ICONS[type];
          return Icon ? (
            <Icon key={type} className="w-3 h-3 text-gray-400" />
          ) : null;
        })}
        {types.length > 3 && (
          <span className="text-gray-400">+{types.length - 3}</span>
        )}
      </div>
    </div>
  );
}
