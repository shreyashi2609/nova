import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler
} from 'chart.js';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Filler);

// --- Types ---
interface Telemetry {
  timestamp?: string;
  device_id?: string;
  status?: 'HEALTHY' | 'DEGRADED' | 'CRITICAL' | string;
  region?: string;
  metrics?: {
    latency_ms?: number;
    packet_loss_pct?: number;
    cpu_util_pct?: number;
  };
}

interface ApprovalRequest {
  thread_id: string;
  action: string;
  risk: number;
  reason: string;
}

const App: React.FC = () => {
  const [telemetry, setTelemetry] = useState<Telemetry[]>([]);
  const [agentLogs, setAgentLogs] = useState<string[]>([]);
  const [pendingApproval, setPendingApproval] = useState<ApprovalRequest | null>(null);
  
  const [chartData, setChartData] = useState({
    labels: [] as string[],
    datasets: [{
      data: [] as number[],
      borderColor: '#00f2ff',
      backgroundColor: 'rgba(0, 242, 255, 0.05)',
      fill: true,
      tension: 0.4,
      pointRadius: 0
    }]
  });

  const telWs = useRef<WebSocket | null>(null);
  const agentWs = useRef<WebSocket | null>(null);

  // 1. Stable Chart Update Function
  const updateChart = useCallback((latency: number) => {
    setChartData(prev => ({
      labels: [...prev.labels, ""].slice(-40),
      datasets: [{
        ...prev.datasets[0],
        data: [...prev.datasets[0].data, latency].slice(-40),
      }]
    }));
  }, []);

  // 2. Main WebSocket Lifecycle
  useEffect(() => {
    let isMounted = true;

    const connect = () => {
      console.log("🔌 Initializing Command Center Streams...");
      
      const tSocket = new WebSocket("wss://nova-backend-abc.onrender.com/ws/telemetry");
      const aSocket = new WebSocket("wss://nova-backend-abc.onrender.com/ws/agent");

      // Telemetry Stream
      tSocket.onmessage = (event) => {
        if (!isMounted) return;
        try {
          const data: Telemetry = JSON.parse(event.data);
          setTelemetry(prev => [data, ...prev].slice(0, 15));
          if (data.metrics?.latency_ms !== undefined) {
            updateChart(data.metrics.latency_ms);
          }
        } catch (err) {
          console.warn("Telemetry stream error", err);
        }
      };

      // Agent Reasoning & HITL Stream
      aSocket.onmessage = (event) => {
        if (!isMounted) return;
        try {
          const msg = JSON.parse(event.data);
          console.log("🤖 Incoming Agent Signal:", msg);

          if (msg.type === "approval_required") {
            setPendingApproval(msg.data);
          } else if (msg.data) {
            // Ensure data is rendered as a string
            const logEntry = typeof msg.data === 'string' ? msg.data : JSON.stringify(msg.data);
            setAgentLogs(prev => [...prev, logEntry].slice(-50));
          }
        } catch (err) {
          console.warn("Agent logic stream error", err);
        }
      };

      tSocket.onopen = () => console.log("✅ Telemetry Link: ESTABLISHED");
      aSocket.onopen = () => console.log("✅ Agent Logic Link: ESTABLISHED");
      
      tSocket.onclose = () => console.log("❌ Telemetry Link: DISCONNECTED");
      aSocket.onclose = () => console.log("❌ Agent Logic Link: DISCONNECTED");

      telWs.current = tSocket;
      agentWs.current = aSocket;
    };

    connect();

    return () => {
      console.log("🧹 Terminating all active streams...");
      isMounted = false;
      telWs.current?.close();
      agentWs.current?.close();
    };
  }, [updateChart]); // Dependency is now stable

  // 3. HITL Approval Handler
  const handleApproval = async (approved: boolean) => {
    if (!pendingApproval) return;
    try {
      await axios.post('http://127.0.0.1:8000/api/approve', {
        approved,
        thread_id: pendingApproval.thread_id
      });
      console.log(`Action ${approved ? 'CONFIRMED' : 'ABORTED'} for ${pendingApproval.thread_id}`);
      setPendingApproval(null);
    } catch (err) {
      console.error("Critical: Failed to transmit approval signal", err);
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    scales: {
      y: { beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
      x: { display: false }
    },
    plugins: { legend: { display: false } }
  };

  const formatTime = (isoString?: string) => {
    if (!isoString || !isoString.includes('T')) return '00:00:00';
    return isoString.split('T')[1].substring(0, 8);
  };

  return (
    <div className="dashboard-root">
      {/* GLOBAL CRITICAL OVERLAY */}
      {telemetry.some(t => t.status === 'CRITICAL') && (
        <div className="critical-overlay">CRITICAL ANOMALY DETECTED</div>
      )}

      {/* HITL MODAL - APPROVAL GATE */}
      {pendingApproval && (
        <div className="hitl-modal">
          <div className="hitl-box">
            <div className="hitl-header">⚠️ INTERVENTION REQUIRED</div>
            <div className="hitl-content">
              <p className="hitl-label">PROPOSED ACTION</p>
              <p className="hitl-value">{pendingApproval.action?.toUpperCase()}</p>
              
              <p className="hitl-label">AI REASONING</p>
              <p className="hitl-reason">{pendingApproval.reason}</p>
              
              <div className="hitl-risk-bar">
                <span>RISK LEVEL: {(pendingApproval.risk * 100).toFixed(0)}%</span>
              </div>
            </div>
            <div className="hitl-buttons">
              <button className="btn-approve" onClick={() => handleApproval(true)}>AUTHORIZE</button>
              <button className="btn-reject" onClick={() => handleApproval(false)}>DENY</button>
            </div>
          </div>
        </div>
      )}

      <header className="top-bar">
        <div className="logo">NEUROTECH // <span>N.O.V.A. COMMAND</span></div>
        <div className="system-status">
          <span className="pulse"></span> 
          CONNECTION: {agentWs.current?.readyState === 1 ? 'ENCRYPTED' : 'OFFLINE'}
        </div>
      </header>

      <div className="main-grid">
        <div className="left-panel">
          <section className="panel graph-box">
            <header className="panel-header">NETWORK LATENCY (MS)</header>
            <div className="content">
              <Line data={chartData} options={chartOptions} />
            </div>
          </section>

          <section className="panel table-box">
            <header className="panel-header">REAL-TIME TELEMETRY FEED</header>
            <div className="content scroll-hide">
              <table>
                <thead>
                  <tr><th>TIMESTAMP</th><th>DEVICE_ID</th><th>STATUS</th><th>LATENCY</th></tr>
                </thead>
                <tbody>
                  {telemetry.map((t, i) => (
                    <tr key={i} className={t.status === 'CRITICAL' ? 'row-critical' : ''}>
                      <td>{formatTime(t.timestamp)}</td>
                      <td>{t.device_id}</td>
                      <td className={`status-cell status-${t.status}`}>{t.status}</td>
                      <td>{t.metrics?.latency_ms}ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>

        <aside className="right-panel">
          <section className="panel log-box">
            <header className="panel-header">NEURAL REASONING ENGINE</header>
            <div className="content log-stream">
              {agentLogs.length === 0 && <div className="log-placeholder">Awaiting neural initialization...</div>}
              {agentLogs.map((log, i) => (
                <div key={i} className="log-entry">
                  <span className="log-cursor">{'>'}</span> {log}
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
};

export default App;