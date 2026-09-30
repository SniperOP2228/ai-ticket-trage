/**
 * Centralized API client for AI Ticket Triage backend service.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function predictTicket(text) {
  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction request failed' }));
    throw new Error(err.detail || 'Failed to predict ticket');
  }
  return res.json();
}

export async function getTickets(params = {}) {
  const query = new URLSearchParams(params).toString();
  const url = `${API_BASE}/api/v1/tickets${query ? `?${query}` : ''}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch tickets');
  return res.json();
}

export async function getTicketById(id) {
  const res = await fetch(`${API_BASE}/api/v1/tickets/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch ticket ${id}`);
  return res.json();
}

export async function getReviewQueue() {
  const res = await fetch(`${API_BASE}/api/v1/review/queue`);
  if (!res.ok) throw new Error('Failed to fetch review queue');
  return res.json();
}

export async function submitReview(ticketId, data) {
  const res = await fetch(`${API_BASE}/api/v1/review/${ticketId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Review submission failed' }));
    throw new Error(err.detail || 'Failed to submit review');
  }
  return res.json();
}

export async function getAnalytics() {
  const res = await fetch(`${API_BASE}/api/v1/analytics`);
  if (!res.ok) throw new Error('Failed to fetch analytics');
  return res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/api/v1/model-info`);
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/v1/health`);
  if (!res.ok) throw new Error('Failed to fetch system health');
  return res.json();
}
