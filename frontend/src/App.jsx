import { useCallback, useEffect, useMemo, useState } from "react";
import toast, { Toaster } from "react-hot-toast";
import { Activity, BarChart3, Brain, GitCompare, LayoutDashboard, Loader2, RotateCcw, Sparkles, Target, Upload } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

const ALGO_OPTIONS = {
  classification: [
    { value: "decision_tree", label: "Decision Tree" },
    { value: "knn", label: "KNN" },
    { value: "naive_bayes", label: "Naive Bayes" },
    { value: "logistic_regression", label: "Logistic Regression" },
    { value: "random_forest_classifier", label: "Random Forest Classifier" },
  ],
  regression: [
    { value: "linear_regression", label: "Linear Regression" },
    { value: "random_forest_regressor", label: "Random Forest Regressor" },
  ],
};

const DEFAULT_ALGO = {
  classification: "decision_tree",
  regression: "linear_regression",
};

function cn(...parts) {
  return parts.filter(Boolean).join(" ");
}

function ParamFields({ algorithm, params, onChange }) {
  if (algorithm === "decision_tree") {
    return (
      <div className="space-y-3">
        <label className="block text-xs font-medium text-slate-600">max_depth
          <input type="number" min={1} max={50} value={params.max_depth ?? 5} onChange={(e) => onChange({ ...params, max_depth: Number(e.target.value) })} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm" />
        </label>
        <label className="block text-xs font-medium text-slate-600">min_samples_split
          <input type="number" min={2} max={100} value={params.min_samples_split ?? 2} onChange={(e) => onChange({ ...params, min_samples_split: Number(e.target.value) })} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm" />
        </label>
      </div>
    );
  }

  if (algorithm === "knn") {
    return (
      <label className="block text-xs font-medium text-slate-600">k (neighbors)
        <input type="number" min={1} max={50} value={params.k ?? 5} onChange={(e) => onChange({ ...params, k: Number(e.target.value) })} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm" />
      </label>
    );
  }

  if (algorithm === "logistic_regression") {
    return (
      <label className="block text-xs font-medium text-slate-600">C (inverse regularization)
        <input type="number" min={0.001} step="0.1" value={params.c ?? 1} onChange={(e) => onChange({ ...params, c: Number(e.target.value) })} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm" />
      </label>
    );
  }

  if (algorithm === "random_forest_classifier" || algorithm === "random_forest_regressor") {
    return (
      <div className="space-y-3">
        <label className="block text-xs font-medium text-slate-600">n_estimators
          <input type="number" min={10} max={1000} value={params.n_estimators ?? 100} onChange={(e) => onChange({ ...params, n_estimators: Number(e.target.value) })} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm" />
        </label>
        <label className="block text-xs font-medium text-slate-600">max_depth (optional)
          <input type="number" min={1} max={100} value={params.max_depth ?? ""} onChange={(e) => onChange({ ...params, max_depth: e.target.value ? Number(e.target.value) : null })} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm" />
        </label>
      </div>
    );
  }

  return <p className="text-xs text-slate-500">No tunable params in this demo.</p>;
}

function ClassificationReportTable({ report }) {
  if (!report || typeof report !== "object") return null;
  const rows = Object.entries(report).filter(([, v]) => v && typeof v === "object" && "precision" in v);
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200">
      <table className="min-w-full text-sm">
        <thead><tr className="border-b bg-slate-50 text-left text-xs uppercase text-slate-500"><th className="px-3 py-2">Class</th><th className="px-3 py-2">Precision</th><th className="px-3 py-2">Recall</th><th className="px-3 py-2">F1</th><th className="px-3 py-2">Support</th></tr></thead>
        <tbody>
          {rows.map(([label, v]) => (
            <tr key={label} className="border-b"><td className="px-3 py-2">{label}</td><td className="px-3 py-2">{Number(v.precision).toFixed(3)}</td><td className="px-3 py-2">{Number(v.recall).toFixed(3)}</td><td className="px-3 py-2">{Number(v["f1-score"]).toFixed(3)}</td><td className="px-3 py-2">{v.support}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MainCharts({ result, taskType, history }) {
  const isCls = taskType === "classification";
  const scoreData = isCls
    ? [{ name: "Accuracy", value: (result.accuracy ?? 0) * 100 }, { name: "Precision", value: (result.precision_score ?? 0) * 100 }, { name: "Recall", value: (result.recall_score ?? 0) * 100 }, { name: "F1", value: (result.f1_score ?? 0) * 100 }]
    : [{ name: "R2", value: result.r2_score ?? 0 }, { name: "MAE", value: result.mae ?? 0 }, { name: "MSE", value: result.mse ?? 0 }];

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="rounded-2xl border border-violet-200/50 bg-white/90 p-4 shadow-lg">
        <p className="mb-2 text-sm font-semibold">{isCls ? "Classification scores" : "Regression scores"}</p>
        <div className="h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={scoreData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="value" radius={[8, 8, 0, 0]}>{scoreData.map((_, i) => <Cell key={i} fill={["#6366f1", "#8b5cf6", "#3b82f6", "#10b981"][i % 4]} />)}</Bar></BarChart></ResponsiveContainer></div>
      </div>

      {history.length > 1 && (
        <div className="rounded-2xl border border-violet-200/50 bg-white/90 p-4 shadow-lg">
          <p className="mb-2 text-sm font-semibold">{isCls ? "Accuracy across runs" : "R2 across runs"}</p>
          <div className="h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={history}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="score" fill="#8b5cf6" radius={[8, 8, 0, 0]} /></BarChart></ResponsiveContainer></div>
        </div>
      )}

      {result.feature_importance?.length > 0 && (
        <div className="rounded-2xl border border-violet-200/50 bg-white/90 p-4 shadow-lg lg:col-span-2">
          <p className="mb-2 text-sm font-semibold">Feature importance</p>
          <div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={[...result.feature_importance].sort((a, b) => b.importance - a.importance).slice(0, 12)} layout="vertical"><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="feature" width={120} /><Tooltip formatter={(v) => Number(v).toFixed(4)} /><Bar dataKey="importance" fill="#6366f1" radius={[0, 8, 8, 0]} /></BarChart></ResponsiveContainer></div>
        </div>
      )}

      {isCls && result.decision_boundary_png && (
        <div className="rounded-2xl border border-violet-200/50 bg-white/90 p-4 shadow-lg lg:col-span-2">
          <p className="mb-2 text-sm font-semibold">Decision boundary (first 2 features)</p>
          <img src={`data:image/png;base64,${result.decision_boundary_png}`} alt="Decision boundary" className="max-h-80 w-full rounded-xl border object-contain" />
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("single");
  const [taskType, setTaskType] = useState("classification");
  const [datasetKey, setDatasetKey] = useState("iris");
  const [csvText, setCsvText] = useState("");
  const [fileName, setFileName] = useState("");
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);

  const [algoSingle, setAlgoSingle] = useState(DEFAULT_ALGO.classification);
  const [paramsSingle, setParamsSingle] = useState({ max_depth: 5, min_samples_split: 2, k: 5, n_estimators: 100, c: 1 });
  const [algoA, setAlgoA] = useState(DEFAULT_ALGO.classification);
  const [paramsA, setParamsA] = useState({ max_depth: 4, min_samples_split: 2, k: 3, n_estimators: 100, c: 1 });
  const [algoB, setAlgoB] = useState("knn");
  const [paramsB, setParamsB] = useState({ max_depth: 5, min_samples_split: 2, k: 5, n_estimators: 100, c: 1 });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [singleResult, setSingleResult] = useState(null);
  const [compareResults, setCompareResults] = useState(null);
  const [bestInfo, setBestInfo] = useState(null);
  const [modelTrained, setModelTrained] = useState(false);
  const [history, setHistory] = useState([]);
  const [runCounter, setRunCounter] = useState(1);
  const [singleInputValues, setSingleInputValues] = useState({});
  const [predictionDisplay, setPredictionDisplay] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);

  const optionsForTask = ALGO_OPTIONS[taskType];

  useEffect(() => {
    setAlgoSingle(DEFAULT_ALGO[taskType]);
    setAlgoA(DEFAULT_ALGO[taskType]);
    setAlgoB(ALGO_OPTIONS[taskType][1]?.value ?? ALGO_OPTIONS[taskType][0].value);
    setSingleResult(null);
    setCompareResults(null);
    setPredictionDisplay(null);
    setHistory([]);
    setRunCounter(1);
  }, [taskType]);

  const datasetPayload = useMemo(() => (csvText ? { dataset: "custom", csv_text: csvText } : { dataset: datasetKey }), [csvText, datasetKey]);

  const refreshModelStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/model/status`);
      const data = await res.json();
      setModelTrained(!!data.trained && data.task_type === taskType);
    } catch {
      setModelTrained(false);
    }
  }, [taskType]);

  useEffect(() => {
    refreshModelStatus();
  }, [refreshModelStatus]);

  useEffect(() => {
    if (singleResult?.feature_names?.length) {
      const init = {};
      singleResult.feature_names.forEach((f) => (init[f] = ""));
      setSingleInputValues(init);
      setPredictionDisplay(null);
    }
  }, [singleResult]);

  const onFile = useCallback((e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFileName(f.name);
    const reader = new FileReader();
    reader.onload = () => setCsvText(String(reader.result ?? ""));
    reader.readAsText(f);
  }, []);

  const buildParams = (algo, p) => {
    if (algo === "decision_tree") return { max_depth: p.max_depth ?? 5, min_samples_split: p.min_samples_split ?? 2 };
    if (algo === "knn") return { k: p.k ?? 5 };
    if (algo === "logistic_regression") return { c: p.c ?? 1 };
    if (algo === "random_forest_classifier" || algo === "random_forest_regressor") return { n_estimators: p.n_estimators ?? 100, max_depth: p.max_depth || null };
    return {};
  };

  const trainSingle = async () => {
    setLoading(true);
    setError("");
    setCompareResults(null);
    setBestInfo(null);
    try {
      const body = { task_type: taskType, algorithm: algoSingle, parameters: buildParams(algoSingle, paramsSingle), test_size: testSize, random_state: randomState, include_boundary: true, ...datasetPayload };
      const res = await fetch(`${API_BASE}/train`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Training failed");
      setSingleResult(data.result);
      setModelTrained(true);
      const score = taskType === "classification" ? (data.result.accuracy ?? 0) * 100 : data.result.r2_score ?? 0;
      setHistory((prev) => [...prev, { name: `Run ${runCounter}`, score }]);
      setRunCounter((c) => c + 1);
      toast.success("Model trained successfully.");
    } catch (err) {
      setError(err.message || String(err));
      setSingleResult(null);
      toast.error(err.message || "Training failed");
    } finally {
      setLoading(false);
    }
  };

  const runCompare = async () => {
    setLoading(true);
    setError("");
    setSingleResult(null);
    try {
      const body = {
        task_type: taskType,
        test_size: testSize,
        random_state: randomState,
        runs: [
          { algorithm: algoA, parameters: buildParams(algoA, paramsA), label: `${optionsForTask.find((o) => o.value === algoA)?.label ?? algoA}` },
          { algorithm: algoB, parameters: buildParams(algoB, paramsB), label: `${optionsForTask.find((o) => o.value === algoB)?.label ?? algoB}` },
        ],
        ...datasetPayload,
      };
      const res = await fetch(`${API_BASE}/compare`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Compare failed");
      setCompareResults(data.results);
      setBestInfo(data.best);
      toast.success("Comparison complete.");
    } catch (err) {
      setError(err.message || String(err));
      setCompareResults(null);
      setBestInfo(null);
      toast.error(err.message || "Comparison failed");
    } finally {
      setLoading(false);
    }
  };

  const predictSingle = async () => {
    if (!modelTrained || !singleResult?.feature_names) return toast.error(`Train a ${taskType} model first.`);
    const nums = [];
    for (const f of singleResult.feature_names) {
      const raw = singleInputValues[f];
      if (raw === "" || raw === undefined) return toast.error(`Fill ${f}`);
      const n = Number(raw);
      if (Number.isNaN(n)) return toast.error(`Invalid ${f}`);
      nums.push(n);
    }
    setPredictLoading(true);
    setPredictionDisplay(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input: nums }) });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Prediction failed");
      setPredictionDisplay(data.prediction);
      toast.success("Prediction ready.");
    } catch (err) {
      toast.error(err.message || "Prediction failed");
    } finally {
      setPredictLoading(false);
    }
  };

  const compareChartData = useMemo(() => {
    if (!compareResults?.length) return [];
    const key = taskType === "classification" ? "accuracy" : "r2_score";
    return compareResults.map((r, i) => ({ name: `M${i + 1}`, label: r.label || r.algorithm, score: taskType === "classification" ? (r[key] ?? 0) * 100 : r[key] ?? 0 }));
  }, [compareResults, taskType]);

  const sidebarCardClass = "rounded-2xl border border-white/40 bg-white/75 p-5 shadow-lg backdrop-blur-md";

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50/40 to-violet-100/70">
      <Toaster position="top-right" toastOptions={{ duration: 3400 }} />

      <header className="sticky top-0 z-40 border-b border-violet-200/40 bg-white/70 backdrop-blur-lg">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-start gap-3"><div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white"><LayoutDashboard className="h-6 w-6" /></div><div><h1 className="text-xl font-bold text-slate-900">Train Sphere</h1><p className="text-sm text-slate-600">Choose task type first, then train with matching algorithms.</p></div></div>
          <div className="flex rounded-xl border border-slate-200/80 bg-white/80 p-1">{[{ id: "single", label: "Single", icon: Brain }, { id: "compare", label: "Compare", icon: GitCompare }].map(({ id, label, icon: Icon }) => (<button key={id} type="button" onClick={() => setMode(id)} className={cn("flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium", mode === id ? "bg-gradient-to-r from-indigo-600 to-violet-600 text-white" : "text-slate-600 hover:bg-slate-100")}><Icon className="h-4 w-4" />{label}</button>))}</div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1600px] flex-col gap-6 px-4 py-8 lg:flex-row lg:items-start lg:px-8">
        <aside className="w-full shrink-0 space-y-5 lg:sticky lg:top-24 lg:w-[360px]">
          <section className={sidebarCardClass}><h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-violet-700">Task type</h2><select value={taskType} onChange={(e) => setTaskType(e.target.value)} className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm"><option value="classification">Classification</option><option value="regression">Regression</option></select></section>
          <section className={sidebarCardClass}><h2 className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-violet-700"><Upload className="h-4 w-4" /> Dataset</h2><label className="block text-xs font-medium text-slate-600">Sample dataset<select value={csvText ? "" : datasetKey} onChange={(e) => { setDatasetKey(e.target.value); setCsvText(""); setFileName(""); }} disabled={!!csvText} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm"><option value="iris">Iris</option><option value="wine">Wine</option></select></label><div className="mt-3"><label className="block text-xs font-medium text-slate-600">Upload CSV</label><input type="file" accept=".csv,text/csv" onChange={onFile} className="mt-1 w-full text-sm file:mr-3 file:rounded-xl file:border-0 file:bg-indigo-600 file:px-3 file:py-2 file:text-white" />{fileName && <p className="mt-1 text-xs text-emerald-600">Loaded: {fileName}</p>}</div></section>
          <section className={sidebarCardClass}><h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-violet-700">Split</h2><div className="grid grid-cols-2 gap-3"><label className="text-xs font-medium text-slate-600">test_size<input type="number" step={0.05} min={0.1} max={0.5} value={testSize} onChange={(e) => setTestSize(Number(e.target.value))} className="mt-1 w-full rounded-xl border border-slate-200 px-2 py-2 text-sm" /></label><label className="text-xs font-medium text-slate-600">random_state<input type="number" value={randomState} onChange={(e) => setRandomState(Number(e.target.value))} className="mt-1 w-full rounded-xl border border-slate-200 px-2 py-2 text-sm" /></label></div></section>

          {mode === "single" ? (
            <section className={sidebarCardClass}><h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-violet-700">Algorithm</h2><label className="mb-3 block text-xs font-medium text-slate-600">Model<select value={algoSingle} onChange={(e) => setAlgoSingle(e.target.value)} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm">{optionsForTask.map((o) => (<option key={o.value} value={o.value}>{o.label}</option>))}</select></label><ParamFields algorithm={algoSingle} params={paramsSingle} onChange={setParamsSingle} /><button type="button" onClick={trainSingle} disabled={loading} className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-3 text-sm font-semibold text-white disabled:opacity-50">{loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}Train model</button><button type="button" onClick={async () => { await fetch(`${API_BASE}/reset`, { method: "POST" }); setModelTrained(false); setSingleResult(null); setPredictionDisplay(null); setHistory([]); setRunCounter(1); toast.success("Model reset."); }} className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white/80 py-2.5 text-sm font-semibold text-slate-700"><RotateCcw className="h-4 w-4" /> Reset model</button></section>
          ) : (
            <>
              <section className={sidebarCardClass}><h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-violet-700">Model A</h2><select value={algoA} onChange={(e) => setAlgoA(e.target.value)} className="mb-3 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm">{optionsForTask.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}</select><ParamFields algorithm={algoA} params={paramsA} onChange={setParamsA} /></section>
              <section className={sidebarCardClass}><h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-violet-700">Model B</h2><select value={algoB} onChange={(e) => setAlgoB(e.target.value)} className="mb-3 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm">{optionsForTask.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}</select><ParamFields algorithm={algoB} params={paramsB} onChange={setParamsB} /></section>
              <button type="button" onClick={runCompare} disabled={loading} className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 py-3 text-sm font-semibold text-white disabled:opacity-50">{loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompare className="h-4 w-4" />} Run comparison</button>
            </>
          )}
        </aside>

        <main className="min-w-0 flex-1 space-y-6">
          {error && <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">{error}</div>}
          {loading && <div className="rounded-2xl border border-violet-200/60 bg-white/80 p-6 shadow-lg"><p className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-700"><Loader2 className="h-4 w-4 animate-spin text-violet-600" /> Training…</p><div className="h-40 rounded-xl bg-slate-100 animate-skeleton" /></div>}

          {!loading && mode === "single" && singleResult && (
            <div className="space-y-6">
              <section className="rounded-2xl border border-violet-200/50 bg-white/90 p-6 shadow-xl"><h2 className="mb-4 flex items-center gap-2 text-lg font-bold"><Activity className="h-5 w-5 text-violet-600" /> Model performance</h2>{taskType === "classification" ? (<div className="grid gap-4 md:grid-cols-4"><div className="rounded-xl bg-indigo-50 p-4"><p className="text-xs">Accuracy</p><p className="text-2xl font-bold">{((singleResult.accuracy ?? 0) * 100).toFixed(2)}%</p></div><div className="rounded-xl bg-violet-50 p-4"><p className="text-xs">Precision</p><p className="text-2xl font-bold">{((singleResult.precision_score ?? 0) * 100).toFixed(2)}%</p></div><div className="rounded-xl bg-blue-50 p-4"><p className="text-xs">Recall</p><p className="text-2xl font-bold">{((singleResult.recall_score ?? 0) * 100).toFixed(2)}%</p></div><div className="rounded-xl bg-emerald-50 p-4"><p className="text-xs">F1</p><p className="text-2xl font-bold">{((singleResult.f1_score ?? 0) * 100).toFixed(2)}%</p></div></div>) : (<div className="grid gap-4 md:grid-cols-3"><div className="rounded-xl bg-indigo-50 p-4"><p className="text-xs">R2 score</p><p className="text-2xl font-bold">{(singleResult.r2_score ?? 0).toFixed(4)}</p></div><div className="rounded-xl bg-violet-50 p-4"><p className="text-xs">MAE</p><p className="text-2xl font-bold">{(singleResult.mae ?? 0).toFixed(4)}</p></div><div className="rounded-xl bg-blue-50 p-4"><p className="text-xs">MSE</p><p className="text-2xl font-bold">{(singleResult.mse ?? 0).toFixed(4)}</p></div></div>)}{taskType === "classification" && (<><h3 className="mb-2 mt-6 text-sm font-semibold">Classification report</h3><ClassificationReportTable report={singleResult.classification_report} /></>)}</section>
              <MainCharts result={singleResult} taskType={taskType} history={history} />
              <section className="rounded-2xl border border-violet-200/50 bg-white/90 p-6 shadow-xl"><h2 className="mb-4 flex items-center gap-2 text-lg font-bold"><Target className="h-5 w-5 text-violet-600" /> Single input prediction</h2><div className="grid gap-3 sm:grid-cols-2">{singleResult.feature_names?.map((fname) => (<label key={fname} className="block text-xs font-medium text-slate-600">{fname}<input type="number" step="any" value={singleInputValues[fname] ?? ""} onChange={(e) => setSingleInputValues((prev) => ({ ...prev, [fname]: e.target.value }))} disabled={!modelTrained} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm disabled:opacity-50" /></label>))}</div><button type="button" onClick={predictSingle} disabled={!modelTrained || predictLoading} className="mt-5 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-3 text-sm font-semibold text-white disabled:opacity-45">{predictLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />} Predict single input</button>{predictionDisplay != null && <div className="mt-6 rounded-2xl border-2 border-emerald-300/80 bg-gradient-to-br from-emerald-50 to-teal-50 p-5 shadow-lg"><p className="text-xs font-bold uppercase text-emerald-800">Prediction</p><p className="mt-1 text-2xl font-bold text-emerald-900">{taskType === "classification" ? "Predicted class" : "Predicted value"}: <span className="text-indigo-700">{String(predictionDisplay)}</span></p></div>}</section>
            </div>
          )}

          {!loading && mode === "compare" && compareResults && (
            <div className="space-y-6">{bestInfo && <div className="rounded-2xl border border-emerald-200/80 bg-gradient-to-r from-emerald-50 to-teal-50 p-5 shadow-lg"><p className="text-sm font-semibold text-emerald-900">Best on this run</p><p className="mt-1 text-lg text-emerald-800">{taskType === "classification" ? `${((bestInfo.score ?? 0) * 100).toFixed(2)}% accuracy` : `${(bestInfo.score ?? 0).toFixed(4)} R2`} — {bestInfo.label}</p></div>}<div className="rounded-2xl border border-violet-200/50 bg-white/90 p-5 shadow-lg"><h3 className="mb-4 flex items-center gap-2 text-base font-semibold"><BarChart3 className="h-4 w-4 text-violet-600" /> Score comparison</h3><div className="h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={compareChartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip formatter={(v, _, ctx) => [v, ctx?.payload?.label || ""]} /><Legend /><Bar dataKey="score" fill="#10b981" name={taskType === "classification" ? "Accuracy %" : "R2"} radius={[8, 8, 0, 0]} /></BarChart></ResponsiveContainer></div></div></div>
          )}

          {!loading && !singleResult && !compareResults && !error && <div className="rounded-2xl border border-dashed border-violet-300/60 bg-white/50 p-12 text-center shadow-inner"><Brain className="mx-auto mb-3 h-10 w-10 text-violet-500 opacity-80" /><p className="text-slate-600">Select <strong>Classification</strong> or <strong>Regression</strong>, then choose matching algorithms and train.</p></div>}
        </main>
      </div>
    </div>
  );
}
