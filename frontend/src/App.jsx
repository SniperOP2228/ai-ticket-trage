import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import DashboardPage from './pages/DashboardPage';
import PredictPage from './pages/PredictPage';
import TicketsPage from './pages/TicketsPage';
import ReviewPage from './pages/ReviewPage';
import AnalyticsPage from './pages/AnalyticsPage';
import ModelInfoPage from './pages/ModelInfoPage';
import { getHealth } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemHealth, setSystemHealth] = useState(null);

  useEffect(() => {
    async function checkHealth() {
      try {
        const health = await getHealth();
        setSystemHealth(health);
      } catch (err) {
        console.warn('API health check error:', err);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col selection:bg-sky-500 selection:text-white">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemHealth={systemHealth}
      />

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        {activeTab === 'dashboard' && <DashboardPage setActiveTab={setActiveTab} />}
        {activeTab === 'predict' && <PredictPage onTicketProcessed={() => {}} />}
        {activeTab === 'tickets' && <TicketsPage />}
        {activeTab === 'review' && <ReviewPage />}
        {activeTab === 'analytics' && <AnalyticsPage />}
        {activeTab === 'model-info' && <ModelInfoPage />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 text-center text-xs text-slate-400 mt-auto">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>AI Customer Support Ticket Triage & Intelligent Routing System</span>
          <span className="font-mono">FastAPI &bull; Scikit-Learn &bull; SentenceTransformers &bull; React &bull; PostgreSQL</span>
        </div>
      </footer>
    </div>
  );
}
