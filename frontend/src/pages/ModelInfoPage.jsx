import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Layers, 
  CheckCircle2, 
  ShieldAlert, 
  BookOpen, 
  FileText,
  Clock,
  Sparkles
} from 'lucide-react';
import { getModelInfo } from '../services/api';

export default function ModelInfoPage() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const info = await getModelInfo();
        setModelInfo(info);
      } catch (err) {
        console.error('Failed to load model info:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const catMeta = modelInfo?.category_model || {};
  const urgMeta = modelInfo?.urgency_model || {};

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
          <Cpu className="w-6 h-6 text-sky-600" />
          Model Governance & Architecture Registry
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Detailed technical specifications, training provenance, validation benchmarks, and confidence calibration bounds.
        </p>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Category Model Card */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div>
              <span className="text-xs uppercase font-bold text-sky-600 tracking-wider">Primary Classifier</span>
              <h3 className="text-lg font-bold text-slate-900">Category Classifier</h3>
            </div>
            <span className="bg-sky-50 text-sky-700 border border-sky-200 text-xs px-2.5 py-1 rounded-full font-mono font-semibold">
              v{modelInfo?.model_version || '1.0.0'}
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Algorithm</span>
              <span className="font-semibold text-slate-800">{catMeta.model_name || 'Random Forest Classifier'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Feature Engineering</span>
              <span className="font-semibold text-slate-800">TF-IDF (10,000 features, Bigrams)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Target Categories</span>
              <span className="font-semibold text-slate-800">{catMeta.n_categories || 10} Classes</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Test Accuracy</span>
              <span className="font-mono font-bold text-emerald-600">
                {catMeta.test_accuracy ? `${(catMeta.test_accuracy * 100).toFixed(2)}%` : '56.88%'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Test Macro F1</span>
              <span className="font-mono font-bold text-emerald-600">
                {catMeta.test_f1_macro ? `${(catMeta.test_f1_macro * 100).toFixed(2)}%` : '59.99%'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Dataset Provenance</span>
              <span className="font-semibold text-slate-800">HF: Tobi-Bueck/customer-support-tickets</span>
            </div>
          </div>
        </div>

        {/* Urgency Model Card */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div>
              <span className="text-xs uppercase font-bold text-indigo-600 tracking-wider">Independent Pipeline</span>
              <h3 className="text-lg font-bold text-slate-900">Urgency Classifier</h3>
            </div>
            <span className="bg-indigo-50 text-indigo-700 border border-indigo-200 text-xs px-2.5 py-1 rounded-full font-mono font-semibold">
              v1.0.0
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Algorithm</span>
              <span className="font-semibold text-slate-800">{urgMeta.model_name || 'Random Forest Classifier'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Feature Engineering</span>
              <span className="font-semibold text-slate-800">TF-IDF (8,000 features, Bigrams)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Target Labels</span>
              <span className="font-semibold text-slate-800">High, Medium, Low</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Test Accuracy</span>
              <span className="font-mono font-bold text-emerald-600">
                {urgMeta.test_accuracy ? `${(urgMeta.test_accuracy * 100).toFixed(2)}%` : '69.62%'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Test Macro F1</span>
              <span className="font-mono font-bold text-emerald-600">
                {urgMeta.test_f1_macro ? `${(urgMeta.test_f1_macro * 100).toFixed(2)}%` : '68.22%'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-50">
              <span className="text-slate-500">Priority Label Source</span>
              <span className="font-semibold text-slate-800">Original Ground Truth Priority</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Selection & Benchmarking Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-4">
        <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-sky-600" />
          Offline Experimentation Benchmarks (Category Classification)
        </h3>
        <p className="text-xs text-slate-500">
          Evaluated across 16,622 training tickets and 3,562 validation tickets with stratified splits. Random Forest provided the highest Macro F1 score on imbalanced categories.
        </p>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-semibold uppercase">
              <tr>
                <th className="px-4 py-3">Candidate Model</th>
                <th className="px-4 py-3">Features</th>
                <th className="px-4 py-3">Validation Accuracy</th>
                <th className="px-4 py-3">Macro F1</th>
                <th className="px-4 py-3">Weighted F1</th>
                <th className="px-4 py-3">Inference Latency</th>
                <th className="px-4 py-3">Decision</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              <tr className="hover:bg-slate-50">
                <td className="px-4 py-3 font-semibold">Logistic Regression</td>
                <td className="px-4 py-3 text-slate-500">TF-IDF (10k)</td>
                <td className="px-4 py-3 font-mono">44.13%</td>
                <td className="px-4 py-3 font-mono">0.4307</td>
                <td className="px-4 py-3 font-mono">0.4433</td>
                <td className="px-4 py-3 font-mono text-emerald-600 font-semibold">3.0 ms</td>
                <td className="px-4 py-3 text-slate-400">Baseline</td>
              </tr>
              <tr className="hover:bg-slate-50">
                <td className="px-4 py-3 font-semibold">Multinomial Naive Bayes</td>
                <td className="px-4 py-3 text-slate-500">TF-IDF (10k)</td>
                <td className="px-4 py-3 font-mono">41.58%</td>
                <td className="px-4 py-3 font-mono">0.3251</td>
                <td className="px-4 py-3 font-mono">0.3891</td>
                <td className="px-4 py-3 font-mono text-emerald-600 font-semibold">3.0 ms</td>
                <td className="px-4 py-3 text-slate-400">Underperformed on minority</td>
              </tr>
              <tr className="hover:bg-slate-50">
                <td className="px-4 py-3 font-semibold">Linear SVM (Calibrated)</td>
                <td className="px-4 py-3 text-slate-500">TF-IDF (10k)</td>
                <td className="px-4 py-3 font-mono">54.01%</td>
                <td className="px-4 py-3 font-mono">0.5274</td>
                <td className="px-4 py-3 font-mono">0.5318</td>
                <td className="px-4 py-3 font-mono">14.5 ms</td>
                <td className="px-4 py-3 text-slate-400">Strong linear runner-up</td>
              </tr>
              <tr className="bg-sky-50/50 hover:bg-sky-50">
                <td className="px-4 py-3 font-bold text-sky-900 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-sky-600" /> Random Forest (Selected)
                </td>
                <td className="px-4 py-3 text-slate-600">TF-IDF (10k, n=200)</td>
                <td className="px-4 py-3 font-mono font-bold text-slate-900">54.66%</td>
                <td className="px-4 py-3 font-mono font-bold text-sky-700">0.5672</td>
                <td className="px-4 py-3 font-mono font-bold text-slate-900">0.5512</td>
                <td className="px-4 py-3 font-mono">89.1 ms</td>
                <td className="px-4 py-3 font-bold text-sky-700">Production Model</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Confidence Threshold & Limitations */}
      <div className="bg-amber-50/80 rounded-2xl border border-amber-200 p-6 space-y-3">
        <h3 className="text-sm font-bold text-amber-900 flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-amber-600" />
          Confidence Scoring Limitations & Human-in-the-Loop Philosophy
        </h3>
        <p className="text-xs text-amber-800 leading-relaxed">
          <strong>Important ML Engineering Note:</strong> Model output probabilities represent relative softmax or tree-ensemble class votes rather than absolute real-world ground truth certainty.
          To mitigate catastrophic routing errors, this system enforces a strict <strong>80% confidence threshold</strong>. Any ticket where either category or urgency confidence is under 80% is withheld from automated delivery and routed to the Human Review Queue.
        </p>
      </div>
    </div>
  );
}
