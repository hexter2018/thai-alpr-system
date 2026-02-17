import React, { useState, useEffect, useCallback, useRef } from 'react';
import StreamViewer from './StreamViewer';
import DetectionOverlay from './DetectionOverlay';
import LogTable from './LogTable';
import AddCameraModal from './Addcameramodal';
import ZoneEditor from './ZoneEditor';
import { useWebSocket } from '../../hooks/useWebSocket';

export default function LiveMonitor() {
  const [cameras, setCameras]               = useState({});
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [loading, setLoading]               = useState(true);
  const [error, setError]                   = useState(null);
  const [actionLoading, setActionLoading]   = useState(false);
  const [showAddModal, setShowAddModal]     = useState(false);
  const [activeTab, setActiveTab]           = useState('monitor'); // 'monitor' | 'zone'
  const [zoneData, setZoneData]             = useState(null);
  const [zoneSaving, setZoneSaving]         = useState(false);
  const [zoneSaved, setZoneSaved]           = useState(false);

  const { messages, isConnected, status, cameraStats, clearMessages } =
    useWebSocket(selectedCamera);

  // ── Load cameras ───────────────────────────────────────────────────────────
  const loadCameras = useCallback(async () => {
    try {
      setError(null);
      const res  = await fetch('/api/stream/list');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const camMap = data.cameras || {};
      setCameras(camMap);
      setSelectedCamera(prev => {
        if (prev && camMap[prev]) return prev;
        const ids = Object.keys(camMap);
        return ids.length > 0 ? ids[0] : null;
      });
    } catch (err) {
      setError('Failed to load cameras: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCameras();
    const t = setInterval(loadCameras, 10_000);
    return () => clearInterval(t);
  }, [loadCameras]);

  // Load zone when camera changes
  useEffect(() => {
    if (!selectedCamera) return;
    setZoneData(null);
    fetch(`/api/stream/zone/${selectedCamera}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setZoneData(d.polygon_zone || []); })
      .catch(() => setZoneData([]));
  }, [selectedCamera]);

  // ── Camera controls ────────────────────────────────────────────────────────
  const handleSelect = (id) => {
    clearMessages();
    setSelectedCamera(id);
    setActiveTab('monitor');
    setZoneSaved(false);
  };

  const handleStart = async (id) => {
    setActionLoading(true);
    try {
      const res  = await fetch(`/api/stream/start/${id}`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Start failed');
      await loadCameras();
    } catch (err) { setError(err.message); }
    finally { setActionLoading(false); }
  };

  const handleStop = async (id) => {
    setActionLoading(true);
    try {
      const res  = await fetch(`/api/stream/stop/${id}`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Stop failed');
      await loadCameras();
    } catch (err) { setError(err.message); }
    finally { setActionLoading(false); }
  };

  const handleCameraAdded = async (id) => {
    setShowAddModal(false);
    await loadCameras();
    setSelectedCamera(id);
  };

  // ── Zone save ──────────────────────────────────────────────────────────────
  const handleSaveZone = async (points) => {
    if (!selectedCamera) return;
    setZoneSaving(true);
    setZoneSaved(false);
    try {
      const res = await fetch(`/api/stream/zone/${selectedCamera}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polygon_zone: points.length >= 3 ? points : null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Zone save failed');
      setZoneData(points.length >= 3 ? points : []);
      setZoneSaved(true);
      setTimeout(() => setZoneSaved(false), 3000);
    } catch (err) {
      setError('Zone save failed: ' + err.message);
    } finally {
      setZoneSaving(false);
    }
  };

  // ── Helpers ────────────────────────────────────────────────────────────────
  const dotColor = {
    connected:    'bg-emerald-400',
    reconnecting: 'bg-amber-400',
    error:        'bg-red-500',
    connecting:   'bg-sky-400',
  }[status] || 'bg-gray-400';

  const camInfo = selectedCamera ? (cameras[selectedCamera] || {}) : {};

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100" style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>

      {/* ══ Top bar ═══════════════════════════════════════════════════════════ */}
      <div className="border-b border-gray-800 bg-gray-900 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sky-400 font-bold tracking-widest text-xs uppercase">ALPR</span>
            <span className="text-gray-600">|</span>
            <span className="text-gray-300 text-sm">Live Monitor</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${dotColor} animate-pulse`} />
            <span className="text-xs text-gray-500 uppercase tracking-wider">{status}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadCameras}
            className="px-3 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-gray-300 transition"
          >
            ⟳ Refresh
          </button>
          <button
            onClick={() => setShowAddModal(true)}
            className="px-3 py-1.5 text-xs bg-sky-600 hover:bg-sky-500 rounded text-white font-medium transition"
          >
            + Add Camera
          </button>
        </div>
      </div>

      <div className="flex h-[calc(100vh-57px)]">

        {/* ══ Left sidebar — camera list ════════════════════════════════════ */}
        <div className="w-56 border-r border-gray-800 bg-gray-900 flex flex-col shrink-0">
          <div className="px-4 py-3 border-b border-gray-800">
            <p className="text-xs text-gray-500 uppercase tracking-widest">Cameras</p>
          </div>

          <div className="flex-1 overflow-y-auto py-2">
            {loading && (
              <p className="px-4 py-6 text-xs text-gray-600 text-center">Loading…</p>
            )}

            {!loading && Object.keys(cameras).length === 0 && (
              <div className="px-4 py-8 text-center">
                <p className="text-gray-500 text-xs mb-3">No cameras configured</p>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="text-xs text-sky-400 hover:text-sky-300"
                >
                  + Add first camera
                </button>
              </div>
            )}

            {Object.entries(cameras).map(([id, info]) => (
              <button
                key={id}
                onClick={() => handleSelect(id)}
                className={`w-full text-left px-4 py-3 border-b border-gray-800/50 transition group ${
                  selectedCamera === id
                    ? 'bg-sky-900/30 border-l-2 border-l-sky-400'
                    : 'hover:bg-gray-800/50 border-l-2 border-l-transparent'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${info.is_running ? 'bg-emerald-400' : 'bg-gray-600'}`} />
                  <span className="text-xs font-medium text-gray-200 truncate">
                    {info.camera_name || id}
                  </span>
                </div>
                <div className="flex items-center justify-between ml-3.5">
                  <span className={`text-[10px] ${info.is_running ? 'text-emerald-500' : 'text-gray-600'}`}>
                    {info.is_running ? `${(info.fps || 0).toFixed(1)} fps` : 'stopped'}
                  </span>
                  {info.has_zone && (
                    <span className="text-[9px] text-sky-500 border border-sky-800 rounded px-1">zone</span>
                  )}
                </div>
              </button>
            ))}
          </div>

          {/* Camera controls */}
          {selectedCamera && cameras[selectedCamera] && (
            <div className="border-t border-gray-800 p-3 space-y-2">
              {cameras[selectedCamera].is_running ? (
                <button
                  onClick={() => handleStop(selectedCamera)}
                  disabled={actionLoading}
                  className="w-full py-1.5 text-xs bg-red-900/50 hover:bg-red-800/60 border border-red-800 rounded text-red-300 transition disabled:opacity-50"
                >
                  ⏹ Stop Stream
                </button>
              ) : (
                <button
                  onClick={() => handleStart(selectedCamera)}
                  disabled={actionLoading}
                  className="w-full py-1.5 text-xs bg-emerald-900/50 hover:bg-emerald-800/60 border border-emerald-800 rounded text-emerald-300 transition disabled:opacity-50"
                >
                  ▶ Start Stream
                </button>
              )}
            </div>
          )}
        </div>

        {/* ══ Main content ══════════════════════════════════════════════════ */}
        <div className="flex-1 flex flex-col overflow-hidden">

          {/* Error banner */}
          {error && (
            <div className="m-4 mb-0 flex justify-between items-center bg-red-950 border border-red-800 rounded px-4 py-2 text-xs text-red-300">
              <span>{error}</span>
              <button onClick={() => setError(null)} className="text-red-500 hover:text-red-300 ml-3">✕</button>
            </div>
          )}

          {selectedCamera ? (
            <>
              {/* Tab bar */}
              <div className="border-b border-gray-800 px-4 flex items-center gap-1 pt-3">
                {[
                  { id: 'monitor', label: '📹 Monitor' },
                  { id: 'zone',    label: '⬡ Zone Editor' },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-4 py-2 text-xs font-medium rounded-t border-b-2 transition ${
                      activeTab === tab.id
                        ? 'text-sky-400 border-sky-400 bg-gray-800/40'
                        : 'text-gray-500 border-transparent hover:text-gray-300'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}

                {/* Stats pills */}
                {cameraStats && (
                  <div className="ml-auto flex items-center gap-2 pb-1">
                    {[
                      { label: 'FPS',    val: (cameraStats.fps || 0).toFixed(1) },
                      { label: 'DETECT', val: cameraStats.detections ?? 0 },
                      { label: 'ERRORS', val: cameraStats.errors ?? 0 },
                    ].map(({ label, val }) => (
                      <div key={label} className="text-center px-2 py-0.5 bg-gray-800 rounded border border-gray-700">
                        <span className="text-xs font-bold text-gray-200">{val}</span>
                        <span className="text-[9px] text-gray-600 ml-1">{label}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Tab: Monitor */}
              {activeTab === 'monitor' && (
                <div className="flex-1 overflow-auto p-4 space-y-4">
                  <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                    <div className="xl:col-span-2">
                      <StreamViewer
                        cameraId={selectedCamera}
                        cameraName={camInfo.camera_name}
                        isRunning={camInfo.is_running}
                      />
                    </div>
                    <DetectionOverlay detections={messages} />
                  </div>
                  <LogTable logs={messages} onClear={clearMessages} />
                </div>
              )}

              {/* Tab: Zone Editor */}
              {activeTab === 'zone' && (
                <div className="flex-1 overflow-auto p-4">
                  <ZoneEditor
                    cameraId={selectedCamera}
                    cameraName={camInfo.camera_name}
                    initialZone={zoneData}
                    onSave={handleSaveZone}
                    saving={zoneSaving}
                    saved={zoneSaved}
                  />
                </div>
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <p className="text-6xl mb-4">📷</p>
                <p className="text-gray-500 text-sm">
                  {loading ? 'Loading cameras…' : 'Select a camera from the left panel'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {showAddModal && (
        <AddCameraModal
          onClose={() => setShowAddModal(false)}
          onAdded={handleCameraAdded}
        />
      )}
    </div>
  );
}