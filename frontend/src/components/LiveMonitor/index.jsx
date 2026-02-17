import React, { useState, useEffect, useCallback } from 'react';
import StreamViewer from './StreamViewer';
import DetectionOverlay from './DetectionOverlay';
import LogTable from './LogTable';
import AddCameraModal from './AddCameraModal';
import { useWebSocket } from '../../hooks/useWebSocket';

export default function LiveMonitor() {
  const [cameras, setCameras]               = useState({});
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [loading, setLoading]               = useState(true);
  const [error, setError]                   = useState(null);
  const [actionLoading, setActionLoading]   = useState(false);
  const [showAddModal, setShowAddModal]     = useState(false);

  const { messages, isConnected, status, cameraStats, clearMessages } =
    useWebSocket(selectedCamera);

  // ── Load camera list ───────────────────────────────────────────────────────
  const loadCameras = useCallback(async () => {
    try {
      setError(null);
      const res  = await fetch('/api/stream/list');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const camMap = data.cameras || {};
      setCameras(camMap);

      setSelectedCamera(prev => {
        if (prev && camMap[prev]) return prev;          // keep valid selection
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

  // ── Camera controls ────────────────────────────────────────────────────────
  const handleSelect = (id) => { clearMessages(); setSelectedCamera(id); };

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

  // ── WS status colour ───────────────────────────────────────────────────────
  const dotColor = { connected:'bg-green-500', reconnecting:'bg-yellow-400',
                     error:'bg-red-500', connecting:'bg-blue-400' }[status] || 'bg-gray-400';

  const camInfo = selectedCamera ? (cameras[selectedCamera] || {}) : {};

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="p-6 space-y-6">

      {/* ─ Header ─ */}
      <div className="flex justify-between items-center flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Live Monitor</h1>
          <p className="text-gray-400 text-sm">Real-time ALPR detection</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`w-2.5 h-2.5 rounded-full ${dotColor}`} />
          <span className="text-sm text-gray-500 capitalize">{status}</span>
          <button onClick={loadCameras}
            className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg">
            🔄 Refresh
          </button>
          <button onClick={() => setShowAddModal(true)}
            className="px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium">
            ➕ Add Camera
          </button>
        </div>
      </div>

      {/* ─ Error ─ */}
      {error && (
        <div className="flex justify-between items-center bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 ml-3">✕</button>
        </div>
      )}

      {/* ─ Camera list ─ */}
      {loading ? (
        <p className="text-gray-400 text-sm">Loading cameras…</p>

      ) : Object.keys(cameras).length === 0 ? (
        /* Empty state */
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 text-center">
          <div className="text-5xl mb-4">📷</div>
          <h2 className="text-lg font-semibold text-gray-800 mb-1">No cameras yet</h2>
          <p className="text-gray-500 text-sm mb-5">
            Add a camera using the button above, or add entries to your <code className="bg-gray-100 px-1 rounded">.env</code> file:
          </p>
          <div className="bg-gray-900 rounded-lg p-4 text-left font-mono text-xs text-green-400 max-w-sm mx-auto text-left">
            <p className="text-gray-500"># In your .env file:</p>
            <p>CAMERA_ID_1=entrance-cam</p>
            <p>RTSP_URL_1=rtsp://admin:pass@IP:554/stream</p>
          </div>
          <button onClick={() => setShowAddModal(true)}
            className="mt-5 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium text-sm">
            ➕ Add First Camera
          </button>
        </div>

      ) : (
        /* Camera selector tabs */
        <div className="flex flex-wrap gap-2 items-center">
          {Object.entries(cameras).map(([id, info]) => (
            <div key={id} className="flex items-center gap-1">
              <button onClick={() => handleSelect(id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition ${
                  selectedCamera === id
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}>
                <span className={`w-2 h-2 rounded-full ${info.is_running ? 'bg-green-400' : 'bg-gray-300'}`} />
                {info.camera_name || id}
              </button>

              {selectedCamera === id && (
                info.is_running
                  ? <button onClick={() => handleStop(id)} disabled={actionLoading} title="Stop"
                      className="px-2 py-2 rounded-lg text-xs bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-50">⏹</button>
                  : <button onClick={() => handleStart(id)} disabled={actionLoading} title="Start"
                      className="px-2 py-2 rounded-lg text-xs bg-green-100 text-green-700 hover:bg-green-200 disabled:opacity-50">▶</button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ─ Stats bar ─ */}
      {cameraStats && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label:'FPS',        val:(cameraStats.fps || 0).toFixed(1) },
            { label:'Frames',     val: cameraStats.frames_processed ?? 0 },
            { label:'Detections', val: cameraStats.detections ?? 0 },
            { label:'Errors',     val: cameraStats.errors ?? 0 },
          ].map(({ label, val }) => (
            <div key={label} className="bg-white rounded-lg border p-3 text-center">
              <p className="text-lg font-bold text-gray-900">{val}</p>
              <p className="text-xs text-gray-400">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* ─ Main view ─ */}
      {selectedCamera && Object.keys(cameras).length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <StreamViewer
                cameraId={selectedCamera}
                cameraName={camInfo.camera_name}
                isRunning={camInfo.is_running}
              />
            </div>
            <DetectionOverlay detections={messages} />
          </div>
          <LogTable logs={messages} onClear={clearMessages} />
        </>
      )}

      {/* ─ Add Camera Modal ─ */}
      {showAddModal && (
        <AddCameraModal
          onClose={() => setShowAddModal(false)}
          onAdded={handleCameraAdded}
        />
      )}
    </div>
  );
}