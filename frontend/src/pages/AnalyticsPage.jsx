import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  PieChart as PieIcon, 
  TrendingUp, 
  RefreshCw, 
  ShieldCheck, 
  CheckCircle2, 
  AlertTriangle,
  UserCheck
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  PieChart, 
  Pie, 
  Cell 
} from 'recharts';
import { getAnalytics } from '../services/api';
import StatCard from '../components/StatCard';

const COLORS = ['#0284c7', '#0d9488', '#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6'];

export default function AnalyticsPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const res = await getAnalytics();
      setData(res);
    } catch (err) {
      console.error('Failed to fetch analytics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  // Format data for Recharts
  const categoryData = data?.category_distribution
    ? Object.entries(data.category_distribution).map(([name, count]) => ({
        name: name.length > 15 ? `${name.substring(0, 15)}...` : name,
        fullName: name,
        tickets: count,
      }))
    : [];

  const urgencyData = data?.urgency_distribution
    ? Object.entries(data.urgency_distribution).map(([name, count]) => ({
        name,
        value: count,
      }))
    : [];

  const confidenceData = data?.confidence_distribution
    ? Object.entries(data.confidence_distribution).map(([bin, count]) => ({
        bin,
        tickets: count,
      }))
    : [];

  const humanReviewRate = data?.total_tickets
    ? (((data.human_review_required + data.reviewed) / data.total_tickets) * 100).toFixed(1)
    : 0;

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-sky-600" />
            Operational & ML Analytics Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Real-time telemetry, routing performance, confidence calibration, and human correction rates.
          </p>
        </div>

        <button
          onClick={fetchAnalytics}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors shadow-2xs"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Metrics
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Tickets"
          value={data?.total_tickets || 0}
          subtitle="Processed via API"
          icon={TrendingUp}
          color="blue"
        />
        <StatCard
          title="Avg Confidence"
          value={data ? `${(data.avg_category_confidence * 100).toFixed(1)}%` : '0%'}
          subtitle="Category classification"
          icon={ShieldCheck}
          color="green"
        />
        <StatCard
          title="Human Review Rate"
          value={`${humanReviewRate}%`}
          subtitle="Low-confidence trigger"
          icon={AlertTriangle}
          color="amber"
        />
        <StatCard
          title="Correction Rate"
          value={data?.human_correction_rate != null ? `${(data.human_correction_rate * 100).toFixed(1)}%` : 'N/A'}
          subtitle="Human overrides on AI"
          icon={UserCheck}
          color="purple"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Category Distribution Chart */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-5">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider mb-4">
            Tickets by Category
          </h3>
          <div className="h-64">
            {categoryData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={categoryData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <XAxis dataKey="name" angle={-35} textAnchor="end" interval={0} tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="tickets" fill="#0284c7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No tickets recorded yet
              </div>
            )}
          </div>
        </div>

        {/* Urgency Distribution Chart */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-5">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider mb-4">
            Urgency / Priority Breakdown
          </h3>
          <div className="h-64 flex items-center justify-center">
            {urgencyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={urgencyData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {urgencyData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-xs text-slate-400">No tickets recorded yet</div>
            )}
          </div>
        </div>

        {/* Confidence Calibration Histogram */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-5 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
                Confidence Score Distribution
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Tickets with confidence below 0.80 automatically enter the Human Review queue.
              </p>
            </div>
            <span className="text-xs bg-slate-100 text-slate-600 px-3 py-1 rounded-full font-semibold">
              Threshold: 80%
            </span>
          </div>

          <div className="h-60">
            {confidenceData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confidenceData} margin={{ top: 10, right: 20, left: -20, bottom: 10 }}>
                  <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="tickets" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No confidence samples yet
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
