"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { BookOpen, Loader2, ArrowLeft, CheckCircle, XCircle, Trophy, RotateCcw, MessageSquare } from "lucide-react";

interface TopicDetail { id: string; name: string; description: string; difficulty: number; keywords: string[]; prerequisites: string[]; }
interface QuizQuestion { id: number; question: string; options: string[]; answer: string; explanation: string; }
type Phase = "learning" | "quiz" | "result";

export default function LearnPage() {
  const params = useParams(); const router = useRouter();
  const topicId = params.topicId as string;
  const [topic, setTopic] = useState<TopicDetail | null>(null);
  const [phase, setPhase] = useState<Phase>("learning");
  const [quiz, setQuiz] = useState<QuizQuestion[]>([]);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [score, setScore] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [busyMsg, setBusyMsg] = useState("");

  useEffect(() => {
    fetch(`http://localhost:8000/api/knowledge-graph/topics/${topicId}`)
      .then((r) => r.json()).then(setTopic).catch(console.error).finally(() => setLoading(false));
  }, [topicId]);

  // 跳转到AI对话页面生成讲解
  const goToChat = () => {
    const msg = `请详细讲解"${topic?.name || topicId}"这个知识点，包含概念定义、关键公式、具体例子和常见误区。Markdown格式输出。`;
    router.push(`/?q=${encodeURIComponent(msg)}`);
  };

  // 生成测试题
  const genQuiz = async () => {
    setBusy(true);
    setBusyMsg("正在生成测试题，请等待30-60秒...");
    try {
      const body = JSON.stringify({ topic_id: topicId, topic_name: topic?.name || topicId, num_questions: 5 });
      console.log("[Quiz] Requesting:", body);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);
      const r = await fetch("http://localhost:8000/api/learn/generate-quiz", {
        method: "POST", headers: { "Content-Type": "application/json" }, body, signal: controller.signal,
      });
      clearTimeout(timeoutId);
      console.log("[Quiz] Status:", r.status);
      if (!r.ok) { const text = await r.text(); throw new Error(`HTTP ${r.status}: ${text}`); }
      const d = await r.json();
      console.log("[Quiz] Data:", d);
      if (d.questions && d.questions.length > 0) {
        setQuiz(d.questions.map((q: QuizQuestion, i: number) => ({ ...q, id: i })));
        setPhase("quiz");
        setBusyMsg("");
      } else {
        setBusyMsg("题目生成失败: " + (d.error || "无题目返回，请重试"));
      }
    } catch (e: unknown) {
      console.error("[Quiz] Error:", e);
      const msg = e instanceof Error ? e.name === "AbortError" ? "请求超时(120秒)，请重试" : e.message : String(e);
      setBusyMsg("生成失败: " + msg);
    }
    setBusy(false);
  };

  const submitQuiz = async () => {
    let correct = 0;
    quiz.forEach((q, i) => { if (answers[i] === q.answer) correct++; });
    const s = Math.round((correct / quiz.length) * 100);
    setScore(s); setPhase("result");
    await fetch("http://localhost:8000/api/progress/quiz-result?user_id=default_user", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic_id: topicId, score: s, total_questions: quiz.length, correct_answers: correct }),
    });
  };

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin text-blue-500" /></div>;
  if (!topic) return <div className="p-6 text-gray-500">知识点不存在</div>;

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-gray-200 bg-white px-6 py-3 flex items-center gap-3">
        <button onClick={() => router.push("/learning-path")} className="p-2 hover:bg-gray-100 rounded-lg"><ArrowLeft className="w-5 h-5 text-gray-500" /></button>
        <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center"><BookOpen className="w-5 h-5 text-white" /></div>
        <div className="flex-1"><h2 className="text-lg font-semibold text-gray-900">{topic.name}</h2><p className="text-xs text-gray-500">难度 {"★".repeat(topic.difficulty)} · {topic.keywords.join(" ")}</p></div>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {phase === "learning" && (
          <div className="max-w-2xl mx-auto space-y-6">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="font-semibold text-gray-800 mb-2">{topic.name}</h3>
              <p className="text-sm text-gray-600 mb-3">{topic.description}</p>
              {topic.prerequisites.length > 0 && <p className="text-xs text-gray-500">前置知识：{topic.prerequisites.join("、")}</p>}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <button onClick={goToChat}
                className="flex items-center justify-center gap-3 p-6 bg-blue-50 hover:bg-blue-100 border-2 border-blue-200 rounded-xl transition-all group">
                <MessageSquare className="w-8 h-8 text-blue-600" />
                <div className="text-left">
                  <p className="text-sm font-semibold text-blue-800">AI生成讲解</p>
                  <p className="text-xs text-blue-600">跳转到AI对话，获取详细讲解</p>
                </div>
              </button>

              <button onClick={genQuiz} disabled={busy}
                className="flex items-center justify-center gap-3 p-6 bg-green-50 hover:bg-green-100 border-2 border-green-200 rounded-xl transition-all disabled:opacity-50">
                {busy ? <Loader2 className="w-8 h-8 text-green-600 animate-spin" /> : <Trophy className="w-8 h-8 text-green-600" />}
                <div className="text-left">
                  <p className="text-sm font-semibold text-green-800">{busy ? "生成中..." : "开始测试"}</p>
                  <p className="text-xs text-green-600">{busy ? busyMsg || "请等待30-60秒" : "5道选择题检验掌握度"}</p>
                </div>
              </button>
            </div>

            {busyMsg && !busy && (
              <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">{busyMsg}</div>
            )}
          </div>
        )}

        {phase === "quiz" && (
          <div className="max-w-2xl mx-auto space-y-6">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="font-semibold text-gray-800 mb-4">掌握度测试 · {topic.name} · {quiz.length}题</h3>
              {quiz.map((q, qi) => (
                <div key={qi} className="mb-6 pb-6 border-b border-gray-100 last:border-0">
                  <p className="text-sm font-medium text-gray-800 mb-3">{qi + 1}. {q.question}</p>
                  <div className="space-y-2">
                    {q.options.map((opt, oi) => (
                      <button key={oi} onClick={() => setAnswers((p) => ({ ...p, [qi]: opt.charAt(0) }))}
                        className={`w-full text-left p-3 rounded-lg border text-sm transition-all ${
                          answers[qi] === opt.charAt(0) ? "border-blue-500 bg-blue-50 text-blue-700" : "border-gray-200 hover:border-gray-300 text-gray-700"
                        }`}>{opt}</button>
                    ))}
                  </div>
                </div>
              ))}
              <button onClick={submitQuiz} disabled={Object.keys(answers).length < quiz.length}
                className="w-full py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white rounded-xl text-sm font-medium">
                提交答案 ({Object.keys(answers).length}/{quiz.length})
              </button>
            </div>
          </div>
        )}

        {phase === "result" && (
          <div className="max-w-2xl mx-auto space-y-6">
            <div className={`rounded-xl p-8 text-center ${score >= 80 ? "bg-green-50 border-2 border-green-300" : "bg-amber-50 border-2 border-amber-300"}`}>
              {score >= 80 ? <Trophy className="w-12 h-12 text-green-600 mx-auto mb-3" /> : <RotateCcw className="w-12 h-12 text-amber-600 mx-auto mb-3" />}
              <p className="text-3xl font-bold text-gray-900 mb-2">{score}分</p>
              <p className="text-sm text-gray-600">{score >= 80 ? "恭喜通过！" : "建议复习后重试"}</p>
            </div>
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-4">答题详情</h3>
              {quiz.map((q, i) => {
                const ok = answers[i] === q.answer;
                return <div key={i} className="mb-4 pb-4 border-b border-gray-100"><div className="flex items-start gap-2">
                  {ok ? <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" /> : <XCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />}
                  <div><p className="text-sm text-gray-800">{i + 1}. {q.question}</p><p className="text-xs text-gray-500 mt-1">你: {answers[i]} · 正确: {q.answer}</p>{!ok && <p className="text-xs text-blue-600 mt-1">解析: {q.explanation}</p>}</div>
                </div></div>;
              })}
            </div>
            <div className="flex gap-3">
              <button onClick={() => { setPhase("learning"); setAnswers({}); setBusyMsg(""); }} className="flex-1 py-3 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-xl text-sm">返回学习</button>
              <button onClick={() => { setQuiz([]); setAnswers({}); genQuiz(); }} className="flex-1 py-3 bg-amber-600 hover:bg-amber-700 text-white rounded-xl text-sm">重新测试</button>
              <button onClick={() => router.push("/learning-path")} className="flex-1 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm">返回路径</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
