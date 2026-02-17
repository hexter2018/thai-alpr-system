import React, { useState, useEffect, useCallback } from 'react';
import StreamViewer from './StreamViewer';
import DetectionOverlay from './DetectionOverlay';
import LogTable from './LogTable';
import { useWebSocket } from '../../hooks/useWebSocket';

// ── Add Camera Modal (inline to avoid missing-file build errors) ─────────────
function AddCameraModal({ onClose, onAdded }) {
  const [form, setForm] = useState({ camera_id: '', camera_name: '', rtsp_url: '', frame_skip: 2 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: name === 'frame_skip' ? Number(value) : value }));
  };

  const handleSubmit = async () => {
    if (!form.camera_id.trim()) return setError('Camera ID is required');
    if (!form.rtsp_url.trim())  return setError('RTSP URL is required');
    setLoading(true); setError(null);
    try {
      const res  = await fetch('/api/stream/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, camera_name: form.camera_name.trim() || form.camera_id }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      onAdded(form.camera_id);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        <div className="flex justify-between items-center px-6 py-4 border-b bg-gray-50">
          <h2 className="text-lg font-semibold text-gray-800">➕ Add Camera</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl">✕</button>
        </div>
        <div className="px-6 py-5 space-y-4">
          {error && <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{error}</div>}

          {[
            { label: 'Camera ID *',   name: 'camera_id',   placeholder: 'PCN_MM04',            hint: 'Unique identifier' },
            { label: 'Camera Name',   name: 'camera_name', placeholder: 'Main Entrance',        hint: 'Display name (optional)' },
            { label: 'RTSP URL *',    name: 'rtsp_url',    placeholder: 'rtsp://user:pass@ip/stream', hint: 'RTSP stream URL or file path', mono: true },
          ].map(({ label, name, placeholder, hint, mono }) => (
            <div key={name}>
              <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
              <input
                name={name} value={form[name]} onChange={handleChange} placeholder={placeholder}
                className={`w-full border border-gray-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 ${mono ? 'font-mono text-xs' : ''}`}
              />
              <p className="text-xs text-gray-400 mt-0.5">{hint}</p>
            </div>
          ))}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Frame Skip: {form.frame_skip}</label>
            <input type="range" name="frame_skip" min={1} max={10} value={form.frame_skip} onChange={handleChange} className="w-full accent-blue-600" />
            <div className="flex justify-between text-xs text-gray-400 mt-0.5"><span>1 (every frame)</span><span>10 (low CPU)</span></div>
          </div>

          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-400 font-medium mb-1">URL formats:</p>
            <div className="space-y-0.5 text-xs font-mono text-gray-500">
              <p>rtsp://admin:pass@<span className="text-blue-500">IP</span>:554/stream1</p>
              <p>rtsp://root:pass@<span className="text-blue-500">IP</span>/axis-media/media.amp</p>
              <p>/app/storage/<span className="text-blue-500">video.mp4</span></p>
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-3 px-6 py-4 border-t bg-gray-50">
          <button onClick={onClose} className="px-4 py-2 text-sm rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-100">Cancel</button>
          <button onClick={handleSubmit} disabled={loading} className="px-5 py-2 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 font-medium">
            {loading ? 'Adding…' : 'Add & Start'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main LiveMonitor ──────────────────────────────────────────────────────────
export default function LiveMonitor() {
  const [cameras, setCameras]               = useState({});
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [loading, setLoading]               = useState(true);
  const [error, setError]                   = useState(null);
  const [actionLoading, setActionLoading]   = useState(false);
  const [showAddModal, setShowAddModal]     = useState(false);

  const { messages, isConnected, status, cameraStats, clearMessages } = useWebSocket(selectedCamera);

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
    } catch (err) { setError('Failed to load cameras: ' + err.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    loadCameras();
    const t = setInterval(loadCameras, 10_000);
    return () => clearInterval(t);
  }, [loadCameras]);

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

  const dotColor = { connected:'bg-green-500', reconnecting:'bg-yellow-400', error:'bg-red-500', connecting:'bg-blue-400' }[status] || 'bg-gray-400';
  const camInfo  = selectedCamera ? (cameras[selectedCamera] || {}) : {};

  return (
    <div className="p-6 space-y-6">

      {/* Header */}
      <div className="flex justify-between items-center flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Live Monitor</h1>
          <p className="text-gray-400 text-sm">Real-time ALPR detection</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`w-2.5 h-2.5 rounded-full ${dotColor}`} />
          <span className="text-sm text-gray-500 capitalize">{status}</span>
          <button onClick={loadCameras} className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg">🔄 Refresh</button>
          <button onClick={() => setShowAddModal(true)} className="px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium">➕ Add Camera</button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex justify-between items-center bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 ml-3">✕</button>
        </div>
      )}

      {/* Camera list */}
      {loading ? (
        <p className="text-gray-400 text-sm">Loading cameras…</p>
      ) : Object.keys(cameras).length === 0 ? (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 text-center">
          <div className="text-5xl mb-4">📷</div>
          <h2 className="text-lg font-semibold text-gray-800 mb-1">No cameras yet</h2>
          <p className="text-gray-500 text-sm mb-4">Add cameras using the button above, or set <code className="bg-gray-100 px-1 rounded">CAMERA_ID_1</code> / <code className="bg-gray-100 px-1 rounded">RTSP_URL_1</code> in your <code className="bg-gray-100 px-1 rounded">.env</code></p>
          <div className="bg-gray-900 rounded-lg p-3 text-left font-mono text-xs text-green-400 max-w-xs mx-auto mb-4">
            <p className="text-gray-500"># .env</p>
            <p>CAMERA_ID_1=PCN_MM04</p>
            <p>RTSP_URL_1=rtsp://...</p>
          </div>
          <button onClick={() => setShowAddModal(true)} className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium text-sm">➕ Add First Camera</button>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2 items-center">
          {Object.entries(cameras).map(([id, info]) => (
            <div key={id} className="flex items-center gap-1">
              <button onClick={() => handleSelect(id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition ${selectedCamera === id ? 'bg-blue-600 text-white shadow-sm' : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'}`}>
                <span className={`w-2 h-2 rounded-full ${info.is_running ? 'bg-green-400' : 'bg-gray-300'}`} />
                {info.camera_name || id}
              </button>
              {selectedCamera === id && (
                info.is_running
                  ? <button onClick={() => handleStop(id)} disabled={actionLoading} title="Stop" className="px-2 py-2 rounded-lg text-xs bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-50">⏹</button>
                  : <button onClick={() => handleStart(id)} disabled={actionLoading} title="Start" className="px-2 py-2 rounded-lg text-xs bg-green-100 text-green-700 hover:bg-green-200 disabled:opacity-50">▶</button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Stats */}
      {cameraStats && (
        <div className="grid grid-cols-4 gap-3">
          {[{l:'FPS',v:(cameraStats.fps||0).toFixed(1)},{l:'Frames',v:cameraStats.frames_processed??0},{l:'Detections',v:cameraStats.detections??0},{l:'Errors',v:cameraStats.errors??0}].map(({l,v})=>(
            <div key={l} className="bg-white rounded-lg border p-3 text-center">
              <p className="text-lg font-bold text-gray-900">{v}</p>
              <p className="text-xs text-gray-400">{l}</p>
            </div>
          ))}
        </div>
      )}

      {/* Main view */}
      {selectedCamera && Object.keys(cameras).length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <StreamViewer cameraId={selectedCamera} cameraName={camInfo.camera_name} isRunning={camInfo.is_running} />
            </div>
            <DetectionOverlay detections={messages} />
          </div>
          <LogTable logs={messages} onClear={clearMessages} />
        </>
      )}

      {/* Modal */}
      {showAddModal && <AddCameraModal onClose={() => setShowAddModal(false)} onAdded={handleCameraAdded} />}
    </div>
  );
}