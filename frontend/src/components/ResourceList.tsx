"use client";

import { useRouter } from "next/navigation";
import { FileText, HelpCircle, Code, Video, Lightbulb, ArrowRight } from "lucide-react";

interface ResourceItem {
  id: string;
  title: string;
  type: string;
  topic: string;
  difficulty: number;
  description: string;
  agent: string;
  duration: string;
  topic_ids?: string[];
}

const TYPE_CONFIG: Record<string, { icon: React.ComponentType<{ className?: string }>; label: string; color: string; bg: string }> = {
  document: { icon: FileText, label: "课程文档", color: "text-blue-600", bg: "bg-blue-50" },
  quiz: { icon: HelpCircle, label: "练习题", color: "text-green-600", bg: "bg-green-50" },
  code: { icon: Code, label: "代码案例", color: "text-purple-600", bg: "bg-purple-50" },
  reading: { icon: Lightbulb, label: "拓展阅读", color: "text-amber-600", bg: "bg-amber-50" },
  multimedia: { icon: Video, label: "多媒体", color: "text-rose-600", bg: "bg-rose-50" },
};

interface ResourceListProps {
  resources: ResourceItem[];
  compact?: boolean;
  maxItems?: number;
  onViewAll?: () => void;
  viewAllLink?: string;
}

export default function ResourceList({
  resources,
  compact = false,
  maxItems,
  onViewAll,
  viewAllLink,
}: ResourceListProps) {
  const router = useRouter();
  const displayed = maxItems ? resources.slice(0, maxItems) : resources;

  if (displayed.length === 0) {
    return (
      <div className="text-center py-4 text-gray-400 text-sm">
        暂无相关资源
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {displayed.map((resource) => {
        const config = TYPE_CONFIG[resource.type] || TYPE_CONFIG.document;
        const Icon = config.icon;

        return (
          <div
            key={resource.id}
            className={`${compact ? "p-2" : "p-3"} rounded-lg border border-gray-100 hover:border-gray-200 hover:bg-gray-50 transition-colors cursor-pointer`}
            onClick={() => {
              if (resource.topic_ids && resource.topic_ids.length > 0) {
                router.push(`/learn/${resource.topic_ids[0]}`);
              }
            }}
          >
            <div className="flex items-start gap-2">
              <div className={`${config.bg} ${compact ? "p-1.5" : "p-2"} rounded-lg flex-shrink-0`}>
                <Icon className={`${compact ? "w-3.5 h-3.5" : "w-4 h-4"} ${config.color}`} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <h4 className={`${compact ? "text-xs" : "text-sm"} font-medium text-gray-900 truncate`}>
                    {resource.title}
                  </h4>
                  {!compact && (
                    <span className="text-xs text-gray-400 flex-shrink-0">{resource.duration}</span>
                  )}
                </div>
                {!compact && (
                  <p className="text-xs text-gray-500 mt-0.5 line-clamp-1">{resource.description}</p>
                )}
                <div className="flex items-center gap-2 mt-1">
                  <span className={`${compact ? "text-[10px]" : "text-xs"} px-1.5 py-0.5 rounded-full ${config.bg} ${config.color}`}>
                    {config.label}
                  </span>
                  {compact && (
                    <span className="text-[10px] text-gray-400">{resource.duration}</span>
                  )}
                  <span className={`${compact ? "text-[10px]" : "text-xs"} text-gray-400`}>
                    难度: {"★".repeat(resource.difficulty)}{"☆".repeat(5 - resource.difficulty)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
      })}

      {(onViewAll || viewAllLink) && resources.length > (maxItems || 0) && (
        <button
          onClick={() => {
            if (onViewAll) {
              onViewAll();
            } else if (viewAllLink) {
              router.push(viewAllLink);
            }
          }}
          className="w-full text-center py-2 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors flex items-center justify-center gap-1"
        >
          查看全部 {resources.length} 份资源
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}
