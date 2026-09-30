import React, { useState } from 'react';
import { 
  Send, 
  Sparkles, 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  Layers, 
  Tag, 
  ArrowRight,
  RotateCcw,
  Check
} from 'lucide-react';
import { predictTicket } from '../services/api';

const SAMPLE_TICKETS = [
  {
    label: "Payment Failure (Billing)",
    text: "My payment was deducted twice for the annual subscription, but my account still says expired. Please refund the duplicate transaction immediately."
  },
  {
    label: "Server 500 Outage (High Urgency)",
    text: "CRITICAL: The production API is returning 500 Internal Server Error for all customer checkout requests. Urgent fix required right now!"
  },
  {
    label: "Return / Refund Request",
    text: "I received the package yesterday but the item was damaged during shipping. How can I initiate a return and get an exchange?"
  },
  {
    label: "Account Password Issue (IT)",
    text: "I forgot my two-factor authentication recovery codes and I am completely locked out of my corporate email account."
  },
  {
    label: "Ambiguous Ticket (Triggers Review)",
    text: "Hello, could you help me with something? Thanks."
  }
];

export default function PredictPage({ onTicketProcessed }) {
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handlePredict = async (e) => {
    e?.preventDefault();
    if (!inputText.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const data = await predictTicket(inputText);
      setResult(data);
      if (onTicketProcessed) onTicketProcessed(data);
    } catch (err) {
      setError(err.message || 'Error processing ticket');
    } finally {
      setLoading(false);
    }
  };

  const getUrgencyBadge = (urgency) => {
    switch (urgency?.toLowerCase()) {
      case 'high':
        return 'bg-rose-100 text-rose-800 border-rose-200';
      case 'medium':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      case 'low':
        return 'bg-emerald-100 text-emerald-800 border-emerald-200';
      default:
        return 'bg-slate-100 text-slate-800 border-slate-200';
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-12">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-sky-600" />
          Customer Support Ticket Triage & Inference
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Input an unstructured customer support ticket to automatically classify category, predict urgency, compute confidence, and determine the optimal routing decision.
        </p>
      </div>

      {/* Preset Ticket Samples */}
      <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
        <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider block mb-2">
          Test With Real Sample Tickets:
        </span>
        <div className="flex flex-wrap gap-2">
          {SAMPLE_TICKETS.map((sample, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setInputText(sample.text);
                setResult(null);
                setError(null);
              }}
              className="text-xs bg-white hover:bg-sky-50 text-slate-700 hover:text-sky-700 border border-slate-200 hover:border-sky-300 rounded-lg px-3 py-1.5 transition-colors font-medium shadow-2xs"
            >
              {sample.label}
            </button>
          ))}
        </div>
      </div>

      {/* Input Form Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6">
        <form onSubmit={handlePredict} className="space-y-4">
          <div>
            <label className="block text-sm font-semibold text-slate-800 mb-1">
              Customer Support Ticket Message
            </label>
            <textarea
              rows={4}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="e.g., My payment was deducted twice for order #49281 but order status still shows cancelled. Need immediate refund!"
              className="w-full rounded-xl border border-slate-300 p-4 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition-all text-slate-900 placeholder:text-slate-400"
              required
            />
            <div className="flex justify-between items-center mt-2 text-xs text-slate-400">
              <span>{inputText.length} characters</span>
              <span>Min 3 characters required</span>
            </div>
          </div>

          <div className="flex items-center justify-between pt-2">
            <button
              type="button"
              onClick={() => {
                setInputText('');
                setResult(null);
                setError(null);
              }}
              disabled={!inputText && !result}
              className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-slate-600 hover:text-slate-900 disabled:opacity-40 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Clear
            </button>

            <button
              type="submit"
              disabled={loading || !inputText.trim()}
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 text-white rounded-xl text-sm font-semibold shadow-xs disabled:opacity-50 transition-all hover:shadow-sm"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Triaging with ML...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Analyze & Route Ticket
                </>
              )}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 text-rose-500 mt-0.5" />
            <div>
              <p className="font-semibold">Inference Error</p>
              <p>{error}</p>
            </div>
          </div>
        )}
      </div>

      {/* Inference Output Card */}
      {result && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-md p-6 space-y-6 animate-fadeIn">
          {/* Top Banner: Status & Routing */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-100">
            <div>
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Ticket #{result.ticket_id} Routing Decision
              </span>
              <div className="mt-1 flex items-center gap-3">
                {result.requires_human_review ? (
                  <span className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-bold bg-amber-100 text-amber-800 border border-amber-300">
                    <AlertTriangle className="w-4 h-4 text-amber-600" />
                    Human Review Required
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    Automatically Routed
                  </span>
                )}
                <span className="text-xs text-slate-400">Model {result.model_version}</span>
              </div>
            </div>

            <div className="text-right">
              <span className="text-xs text-slate-500">Destination Queue</span>
              <p className="text-base font-bold text-slate-900 uppercase font-mono tracking-tight text-sky-700 bg-sky-50 px-3 py-1 rounded-lg border border-sky-100 inline-block mt-0.5">
                {result.queue}
              </p>
            </div>
          </div>

          {/* Classification Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Category Card */}
            <div className="bg-slate-50 rounded-xl p-5 border border-slate-200/80">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-sky-600" /> Category
                </span>
                <span className="text-xs font-bold text-slate-700">
                  {(result.category_confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <p className="text-xl font-bold text-slate-900 mt-2">{result.category}</p>

              {/* Confidence Bar */}
              <div className="w-full bg-slate-200 rounded-full h-2 mt-3 overflow-hidden">
                <div 
                  className={`h-2 rounded-full ${
                    result.category_confidence >= 0.8 ? 'bg-emerald-500' : 'bg-amber-500'
                  }`}
                  style={{ width: `${Math.min(result.category_confidence * 100, 100)}%` }}
                />
              </div>
            </div>

            {/* Urgency Card */}
            <div className="bg-slate-50 rounded-xl p-5 border border-slate-200/80">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                  <Clock className="w-4 h-4 text-indigo-600" /> Urgency / Priority
                </span>
                <span className="text-xs font-bold text-slate-700">
                  {(result.urgency_confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <span className={`px-3 py-1 rounded-lg text-sm font-bold border ${getUrgencyBadge(result.urgency)}`}>
                  {result.urgency} Priority
                </span>
              </div>

              {/* Confidence Bar */}
              <div className="w-full bg-slate-200 rounded-full h-2 mt-3 overflow-hidden">
                <div 
                  className={`h-2 rounded-full ${
                    result.urgency_confidence >= 0.8 ? 'bg-emerald-500' : 'bg-amber-500'
                  }`}
                  style={{ width: `${Math.min(result.urgency_confidence * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Explainable AI: Important Linguistic Features */}
          {result.important_features && result.important_features.length > 0 && (
            <div className="pt-2">
              <div className="flex items-center gap-2 mb-2">
                <Tag className="w-4 h-4 text-sky-600" />
                <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Explainable AI (Important Linguistic Features)
                </span>
              </div>
              <p className="text-xs text-slate-500 mb-3">
                Key model vocabulary tokens that correlated with this category classification (TF-IDF feature weights):
              </p>
              <div className="flex flex-wrap gap-2">
                {result.important_features.map((feature, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 bg-sky-50 text-sky-800 border border-sky-200 px-3 py-1 rounded-md text-xs font-medium font-mono"
                  >
                    #{feature}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
