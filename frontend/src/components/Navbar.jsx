import React from 'react';
import { 
  LayoutDashboard, 
  Send, 
  Inbox, 
  UserCheck, 
  BarChart3, 
  Cpu, 
  CheckCircle2, 
  AlertCircle 
} from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, systemHealth }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'predict', label: 'Predict & Route', icon: Send },
    { id: 'tickets', label: 'Tickets History', icon: Inbox },
    { id: 'review', label: 'Human Review', icon: UserCheck },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'model-info', label: 'Model Info', icon: Cpu },
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-sky-600 to-indigo-600 text-white p-2.5 rounded-xl shadow-sm">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-lg text-slate-900 tracking-tight flex items-center gap-2">
                AutoTriage AI
                <span className="text-[10px] uppercase font-semibold bg-sky-100 text-sky-700 px-2 py-0.5 rounded-full border border-sky-200">
                  Production v1.0
                </span>
              </span>
              <p className="text-xs text-slate-500 hidden sm:block">Intelligent Support Routing & Human-in-the-Loop</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-2 bg-slate-50 px-3 py-1.5 rounded-full border border-slate-200 text-xs">
              <span className="text-slate-500">Service:</span>
              {systemHealth?.status === 'healthy' ? (
                <span className="inline-flex items-center gap-1 font-medium text-emerald-600">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Healthy
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 font-medium text-amber-600">
                  <AlertCircle className="w-3.5 h-3.5" /> Connecting
                </span>
              )}
              <span className="text-slate-300">|</span>
              <span className="text-slate-500">Threshold:</span>
              <span className="font-semibold text-slate-700">{(systemHealth?.confidence_threshold * 100) || 80}%</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 sm:space-x-4 overflow-x-auto py-2 border-t border-slate-100">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-sky-50 text-sky-700 shadow-xs border border-sky-100'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-sky-600' : 'text-slate-400'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
