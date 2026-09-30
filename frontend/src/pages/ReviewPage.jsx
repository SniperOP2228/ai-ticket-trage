import React, { useState, useEffect } from 'react';
import { 
  UserCheck, 
  AlertTriangle, 
  CheckCircle2, 
  RefreshCw, 
  Send, 
  Sparkles,
  ArrowRight
} from 'lucide-react';
import { getReviewQueue, submitReview } from '../services/api';

const CATEGORIES = [
  "Billing and Payments",
  "Customer Service",
  "General Inquiry",
  "Human Resources",
  "IT Support",
  "Product Support",
  "Returns and Exchanges",
  "Sales and Pre-Sales",
  "Service Outages and Maintenance",
  "Technical Support"
];

const URGENCIES = ["Low", "Medium", "High"];

export default function ReviewPage() {
  const [queue, setQueue] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [activeTicket, setActiveTicket] = useState(null);
  const [correctedCategory, setCorrectedCategory] = useState('');
  const [correctedUrgency, setCorrectedUrgency] = useState('');
  const [reviewerName, setReviewerName] = useState('Senior Support Agent');
  const [notes, setNotes] = useState('');
  const [successMessage, setSuccessMessage] = useState(null);

  const fetchQueue = async () => {
    setLoading(true);
    try {
      const data = await getReviewQueue();
      setQueue(data || []);
      if (data && data.length > 0 && !activeTicket) {
        selectTicket(data[0]);
      } else if (!data || data.length === 0) {
        setActiveTicket(null);
      }
    } catch (err) {
      console.error('Failed to fetch review queue:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, []);

  const selectTicket = (t) => {
    setActiveTicket(t);
    setCorrectedCategory(t.predicted_category || CATEGORIES[0]);
    setCorrectedUrgency(t.predicted_urgency || URGENCIES[1]);
    setNotes('');
    setSuccessMessage(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!activeTicket) return;

    setSubmitting(true);
    try {
      await submitReview(activeTicket.ticket_id, {
        corrected_category: correctedCategory,
        corrected_urgency: correctedUrgency,
        reviewer: reviewerName,
        notes: notes.trim() || undefined
      });

      setSuccessMessage(`Ticket #${activeTicket.ticket_id} reviewed and updated successfully!`);
      // Remove from current queue
      const updatedQueue = queue.filter(q => q.ticket_id !== activeTicket.ticket_id);
      setQueue(updatedQueue);
      if (updatedQueue.length > 0) {
        selectTicket(updatedQueue[0]);
      } else {
        setActiveTicket(null);
      }
    } catch (err) {
      console.error('Failed to submit review:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <UserCheck className="w-6 h-6 text-sky-600" />
            Human-in-the-Loop Review Queue
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Review and correct tickets where the AI model confidence fell below the configured threshold (80%).
          </p>
        </div>

        <button
          onClick={fetchQueue}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors shadow-2xs"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Queue
        </button>
      </div>

      {successMessage && (
        <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 p-4 rounded-xl text-xs font-medium flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
          <span>{successMessage}</span>
        </div>
      )}

      {/* Review Workflow Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Tickets Queue List (5 cols) */}
        <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200 shadow-xs p-4 flex flex-col h-[600px]">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
              Pending Tickets ({queue.length})
            </span>
            <span className="text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full font-semibold border border-amber-200">
              Needs Human Review
            </span>
          </div>

          <div className="overflow-y-auto flex-1 mt-3 space-y-2 pr-1">
            {loading && queue.length === 0 ? (
              <div className="text-center py-12 text-slate-400 text-xs">
                <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-sky-600" />
                Loading review queue...
              </div>
            ) : queue.length === 0 ? (
              <div className="text-center py-16 text-slate-400 text-xs">
                <CheckCircle2 className="w-8 h-8 text-emerald-500 mx-auto mb-2" />
                <p className="font-semibold text-slate-700">All clear!</p>
                <p className="mt-1">No tickets currently pending human review.</p>
              </div>
            ) : (
              queue.map((t) => {
                const isSelected = activeTicket?.ticket_id === t.ticket_id;
                return (
                  <div
                    key={t.ticket_id}
                    onClick={() => selectTicket(t)}
                    className={`p-3.5 rounded-xl border text-xs cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-sky-50/80 border-sky-300 shadow-xs'
                        : 'bg-white hover:bg-slate-50 border-slate-200'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-mono font-bold text-slate-500">#{t.ticket_id}</span>
                      <span className="bg-amber-50 text-amber-700 border border-amber-200 px-2 py-0.5 rounded text-[10px] font-bold">
                        {(t.category_confidence * 100).toFixed(0)}% conf
                      </span>
                    </div>
                    <p className="text-slate-800 font-medium line-clamp-2">{t.text}</p>
                    <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                      <span>Pred: <strong className="text-slate-600">{t.predicted_category}</strong></span>
                      <span>{new Date(t.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Right Column: Review & Correction Form (7 cols) */}
        <div className="lg:col-span-7 bg-white rounded-2xl border border-slate-200 shadow-xs p-6 flex flex-col justify-between">
          {activeTicket ? (
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="flex items-center justify-between pb-4 border-b border-slate-100">
                <div>
                  <span className="text-xs font-mono font-bold text-sky-600">Reviewing Ticket #{activeTicket.ticket_id}</span>
                  <h3 className="text-lg font-bold text-slate-900 mt-0.5">Inspect & Submit Correction</h3>
                </div>
                <div className="text-right">
                  <span className="text-xs text-slate-400 block">AI Confidence</span>
                  <span className="text-sm font-bold text-amber-600">
                    {(activeTicket.category_confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Original Ticket Text */}
              <div>
                <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider block mb-1">
                  Customer Message:
                </label>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs text-slate-800 leading-relaxed font-sans max-h-40 overflow-y-auto whitespace-pre-wrap">
                  {activeTicket.text}
                </div>
              </div>

              {/* AI Prediction vs Human Correction */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* AI Suggestions */}
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-2 text-xs">
                  <span className="font-semibold text-slate-500 uppercase tracking-wider block">AI Prediction</span>
                  <div>
                    <span className="text-slate-400 block text-[11px]">Category:</span>
                    <span className="font-semibold text-slate-800">{activeTicket.predicted_category}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[11px]">Urgency:</span>
                    <span className="font-semibold text-slate-800">{activeTicket.predicted_urgency}</span>
                  </div>
                </div>

                {/* Human Form Fields */}
                <div className="space-y-3 text-xs">
                  <div>
                    <label className="font-semibold text-slate-700 block mb-1">Correct Category</label>
                    <select
                      value={correctedCategory}
                      onChange={(e) => setCorrectedCategory(e.target.value)}
                      className="w-full bg-white border border-slate-300 rounded-lg p-2 font-medium text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-500"
                    >
                      {CATEGORIES.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="font-semibold text-slate-700 block mb-1">Correct Urgency</label>
                    <select
                      value={correctedUrgency}
                      onChange={(e) => setCorrectedUrgency(e.target.value)}
                      className="w-full bg-white border border-slate-300 rounded-lg p-2 font-medium text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-500"
                    >
                      {URGENCIES.map((u) => (
                        <option key={u} value={u}>{u}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Reviewer Meta */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div>
                  <label className="font-semibold text-slate-700 block mb-1">Reviewer Name / ID</label>
                  <input
                    type="text"
                    value={reviewerName}
                    onChange={(e) => setReviewerName(e.target.value)}
                    className="w-full bg-white border border-slate-300 rounded-lg p-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-500"
                    required
                  />
                </div>
                <div>
                  <label className="font-semibold text-slate-700 block mb-1">Review Notes (Optional)</label>
                  <input
                    type="text"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="e.g., Customer confirmed billing dispute"
                    className="w-full bg-white border border-slate-300 rounded-lg p-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-500"
                  />
                </div>
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  type="submit"
                  disabled={submitting}
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-sky-600 hover:bg-sky-700 text-white rounded-xl text-xs font-semibold shadow-xs disabled:opacity-50 transition-colors"
                >
                  {submitting ? (
                    'Saving Correction...'
                  ) : (
                    <>
                      <Send className="w-3.5 h-3.5" />
                      Submit Review & Route
                    </>
                  )}
                </button>
              </div>
            </form>
          ) : (
            <div className="h-full flex items-center justify-center text-center py-20 text-slate-400 text-xs">
              <p>Select a ticket from the queue on the left to start review.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
