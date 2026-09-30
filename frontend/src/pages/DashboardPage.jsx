import React, { useState, useEffect } from 'react';
import { 
  Inbox, 
  Send, 
  CheckCircle2, 
  AlertTriangle, 
  TrendingUp, 
  Clock, 
  Layers, 
  ArrowRight,
  ShieldCheck,
  UserCheck,
  RefreshCw,
  Sparkles
} from 'lucide-react';
import StatCard from '../components/StatCard';
import { getAnalytics, getTickets } from '../services/api';

export default function DashboardPage({ setActiveTab }) {
  const [analytics, setAnalytics] = useState(null);
  const [recentTickets, setRecentTickets] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [stats, ticketsData] = await Promise.all([
        getAnalytics(),
        getTickets({ page: 1, page_size: 5 })
      ]);
      setAnalytics(stats);
      setRecentTickets(ticketsData.tickets || []);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const autoRoutedPct = analytics?.total_tickets
    ? ((analytics.auto_routed / analytics.total_tickets) * 100).toFixed(0)
    : '0';

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      {/* Hero Welcome Banner */}
      <div className="bg-gradient-to-r from-sky-900 via-indigo-900 to-slate-900 text-white rounded-3xl p-8 shadow-sm relative overflow-hidden">
        <div className="relative z-10 max-w-2xl space-y-3">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-sky-500/20 text-sky-200 border border-sky-400/30">
            <Sparkles className="w-3.5 h-3.5" /> Production AI/ML Triage Architecture
          </span>
          <h1 className="text-3xl font-extrabold tracking-tight">
            Customer Support Ticket Triage & Intelligent Routing
          </h1>
          <p className="text-sm text-slate-300 leading-relaxed">
            Real-time multi-task NLP classifying ticket category and urgency, backed by calibrated confidence scoring, explainable linguistic features, and a closed-loop human review feedback cycle.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => setActiveTab('predict')}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-xs font-bold transition-colors shadow-xs"
            >
              <Send className="w-3.5 h-3.5" /> Analyze New Ticket
            </button>
            <button
              onClick={() => setActiveTab('review')}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl text-xs font-semibold border border-white/15 transition-colors"
            >
              <UserCheck className="w-3.5 h-3.5" /> Human Review Queue ({analytics?.human_review_required || 0})
            </button>
          </div>
        </div>

        {/* Decorative background glow */}
        <div className="absolute right-0 top-0 bottom-0 w-96 bg-gradient-to-l from-sky-500/10 to-transparent pointer-events-none" />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Processed"
          value={analytics?.total_tickets || 0}
          subtitle="Tickets routed to queues"
          icon={Inbox}
          color="blue"
        />
        <StatCard
          title="Auto-Routed"
          value={`${autoRoutedPct}%`}
          subtitle={`${analytics?.auto_routed || 0} tickets high confidence`}
          icon={CheckCircle2}
          color="green"
        />
        <StatCard
          title="Needs Review"
          value={analytics?.human_review_required || 0}
          subtitle="Confidence below 80%"
          icon={AlertTriangle}
          color="amber"
        />
        <StatCard
          title="High Priority"
          value={analytics?.high_urgency_count || 0}
          subtitle="Critical urgency tickets"
          icon={Clock}
          color="red"
        />
      </div>

      {/* Two Column Layout: Architecture Flow + Recent Tickets */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Pipeline Architecture Steps (5 cols) */}
        <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-4">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-600" />
            Decision Flow Architecture
          </h3>
          <p className="text-xs text-slate-500">
            Every incoming ticket executes an asynchronous multi-stage evaluation pipeline:
          </p>

          <div className="space-y-3 pt-2 text-xs">
            <div className="flex gap-3 items-start">
              <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-700 font-bold flex items-center justify-center flex-shrink-0 text-[11px]">1</span>
              <div>
                <strong className="text-slate-800 block">Text Normalization & NLP Extraction</strong>
                <span className="text-slate-500">Subject/body merged, URL/email stripped, vocabulary tokenized.</span>
              </div>
            </div>

            <div className="flex gap-3 items-start">
              <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-700 font-bold flex items-center justify-center flex-shrink-0 text-[11px]">2</span>
              <div>
                <strong className="text-slate-800 block">Multi-Task Model Inference</strong>
                <span className="text-slate-500">Category (10 classes) and Urgency (3 tiers) models predict simultaneously.</span>
              </div>
            </div>

            <div className="flex gap-3 items-start">
              <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-700 font-bold flex items-center justify-center flex-shrink-0 text-[11px]">3</span>
              <div>
                <strong className="text-slate-800 block">Confidence Threshold Gating (80%)</strong>
                <span className="text-slate-500">Scores &ge; 0.80 auto-route directly to specialized queues; below 0.80 routes to humans.</span>
              </div>
            </div>

            <div className="flex gap-3 items-start">
              <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-700 font-bold flex items-center justify-center flex-shrink-0 text-[11px]">4</span>
              <div>
                <strong className="text-slate-800 block">Human Feedback Loop & Persistence</strong>
                <span className="text-slate-500">Corrections logged in database for offline active learning and continuous retraining.</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Recent Tickets Activity Feed (7 cols) */}
        <div className="lg:col-span-7 bg-white rounded-2xl border border-slate-200 shadow-xs p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                Recent Triage Activity
              </h3>
              <button
                onClick={() => setActiveTab('tickets')}
                className="text-xs font-semibold text-sky-600 hover:text-sky-800 flex items-center gap-1"
              >
                View All <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="mt-3 divide-y divide-slate-100">
              {loading && recentTickets.length === 0 ? (
                <div className="text-center py-10 text-slate-400 text-xs">
                  <RefreshCw className="w-4 h-4 animate-spin mx-auto mb-2 text-sky-600" />
                  Loading recent tickets...
                </div>
              ) : recentTickets.length === 0 ? (
                <div className="text-center py-12 text-slate-400 text-xs">
                  No tickets processed yet. Click &quot;Analyze New Ticket&quot; above to get started!
                </div>
              ) : (
                recentTickets.map((t) => (
                  <div key={t.id} className="py-3 flex items-start justify-between gap-4 text-xs">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-slate-400 font-bold">#{t.id}</span>
                        <span className="font-semibold text-slate-800">{t.category || 'Pending'}</span>
                        <span className="text-slate-300">&bull;</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                          t.urgency?.toLowerCase() === 'high' ? 'bg-rose-100 text-rose-700' :
                          t.urgency?.toLowerCase() === 'medium' ? 'bg-amber-100 text-amber-700' :
                          'bg-emerald-100 text-emerald-700'
                        }`}>
                          {t.urgency || 'Normal'}
                        </span>
                      </div>
                      <p className="text-slate-600 line-clamp-1 max-w-md">{t.text}</p>
                    </div>

                    <div className="text-right flex-shrink-0">
                      <span className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        t.status === 'routed' ? 'bg-emerald-100 text-emerald-800' :
                        t.status === 'review_required' ? 'bg-amber-100 text-amber-800' :
                        'bg-sky-100 text-sky-800'
                      }`}>
                        {t.status === 'routed' ? 'Auto-Routed' :
                         t.status === 'review_required' ? 'Needs Review' : 'Reviewed'}
                      </span>
                      <span className="block text-[10px] text-slate-400 mt-1">
                        {new Date(t.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="pt-4 border-t border-slate-100 flex justify-end">
            <button
              onClick={() => setActiveTab('predict')}
              className="text-xs font-semibold text-sky-600 hover:text-sky-800 flex items-center gap-1"
            >
              Submit another ticket &rarr;
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
