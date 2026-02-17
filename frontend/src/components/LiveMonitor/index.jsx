import React, { useState, useEffect, useCallback } from 'react';
import StreamViewer from './StreamViewer';
import DetectionOverlay from './DetectionOverlay';
import LogTable from './LogTable';
import { useWebSocket } from '../../hooks/useWebSocket';

export default function LiveMonitor() {
  // cameras is an OBJECT keyed by camera_id (matches API response)
  const [cameras, setCameras]           = useState({});
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);
  const [actionLoading, setActionLoading] = useState(false);

  const { messages, isConnected, status, cameraStats, clearMessages } =
    useWebSocket(selectedCamera);

  // ── Load / refresh camera list ─────────────────────────────────────────────
  const loadCameras = useCallback(async () => {
    try {
      setError(null);
      const res  = await fetch('/api/stream/list');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const camMap = data.cameras || {};
      setCameras(camMap);

      // Auto-select first camera on initial load
      setSelectedCamera((prev) => {
        if (prev) return prev;           // keep existing selection
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
    const interval = setInterval(loadCameras, 10_000);
    return () => clearInterval(interval);
  }, [loadCameras]);

  // ── Camera controls ────────────────────────────────────────────────────────
  const handleSelect = (camId) => {
    clearMessages();
    setSelectedCamera(camId);
  };

  const handleStart = async (camId) => {
    setActionLoading(true);
    try {
      const res  = await fetch(`/api/stream/start/${camId}`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to start');
      await loadCameras();
    } catch (err) {
      setError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleStop = async (camId) => {
    setActionLoading(true);
    try {
      const res = await fetch(`/api/stream/stop/${camId}`, { method: 'POST' });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || 'Failed to stop');
      }
      await loadCameras();
    } catch (err) {
      setError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  // ── Status indicator colour ────────────────────────────────────────────────
  const statusColour = {
    connected:    'bg-green-500',
    reconnecting: 'bg-yellow-500',
    error:        'bg-red-500',
    connecting:   'bg-blue-400',
    disconnected: 'bg-gray-400',
  }[status] || 'bg-gray-400';

  const camInfo = selectedCamera ? cameras[selectedCamera] : null;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="p-6 space-y-6">

      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Live Monitor</h1>
          <p className="text-gray-500 text-sm">Real-time ALPR detection</p>
        </div>
        <div className="flex items-center space-x-3">
          <span className={`w-2.5 h-2.5 rounded-full ${statusColour}`} />
          <span className="text-sm capitalize text-gray-600">{status}</span>
          <button
            onClick={loadCameras}
            className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700 flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600">✕</button>
        </div>
      )}

      {/* Camera selector */}
      {loading ? (
        <p className="text-gray-400 text-sm">Loading cameras…</p>
      ) : Object.keys(cameras).length === 0 ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-800">
          <p className="font-semibold">No cameras found.</p>
          <p className="mt-1">
            Add a camera via the API, or set <code>CAMERA_ID_1</code> / <code>RTSP_URL_1</code> in your <code>.env</code>.
          </p>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2 items-center">
          {Object.entries(cameras).map(([camId, info]) => (
            <div key={camId} className="flex items-center gap-1">
              {/* Select button */}
              <button
                onClick={() => handleSelect(camId)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition ${
                  selectedCamera === camId
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${info.is_running ? 'bg-green-400' : 'bg-gray-300'}`} />
                {info.camera_name || camId}
              </button>

              {/* Start / Stop only for selected camera */}
              {selectedCamera === camId && (
                info.is_running ? (
                  <button
                    onClick={() => handleStop(camId)}
                    disabled={actionLoading}
                    title="Stop camera"
                    className="px-2 py-2 rounded-lg text-xs bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-50"
                  >
                    ⏹
                  </button>
                ) : (
                  <button
                    onClick={() => handleStart(camId)}
                    disabled={actionLoading}
                    title="Start camera"
                    className="px-2 py-2 rounded-lg text-xs bg-green-100 text-green-700 hover:bg-green-200 disabled:opacity-50"
                  >
                    ▶
                  </button>
                )
              )}
            </div>
          ))}
        </div>
      )}

      {/* Stats bar */}
      {cameraStats && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'FPS',        value: (cameraStats.fps || 0).toFixed(1) },
            { label: 'Frames',     value: cameraStats.frames_processed ?? 0 },
            { label: 'Detections', value: cameraStats.detections ?? 0 },
            { label: 'Errors',     value: cameraStats.errors ?? 0 },
          ].map(({ label, value }) => (
            <div key={label} className="bg-white rounded-lg border p-3 text-center">
              <p className="text-lg font-bold text-gray-900">{value}</p>
              <p className="text-xs text-gray-400">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Main content */}
      {selectedCamera ? (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <StreamViewer
                cameraId={selectedCamera}
                cameraName={camInfo?.camera_name}
                isRunning={camInfo?.is_running}
              />
            </div>
            <div>
              <DetectionOverlay detections={messages} />
            </div>
          </div>

          <LogTable logs={messages} onClear={clearMessages} />
        </>
      ) : (
        <div className="bg-gray-50 rounded-xl p-16 text-center text-gray-400">
          Select a camera above to start monitoring
        </div>
      )}

    </div>
  );
}