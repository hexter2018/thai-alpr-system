import React, { useState } from 'react';

export default function AddCameraModal({ onClose, onAdded }) {
  const [form, setForm] = useState({
    camera_id: '',
    camera_name: '',
    rtsp_url: '',
    frame_skip: 2,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: name === 'frame_skip' ? Number(value) : value }));
  };

  const handleSubmit = async () => {
    if (!form.camera_id.trim()) return setError('Camera ID is required');
    if (!form.rtsp_url.trim())  return setError('RTSP URL is required');

    setLoading(true);
    setError(null);
    try {
      const res  = await fetch('/api/stream/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          camera_name: form.camera_name.trim() || form.camera_id,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      onAdded(form.camera_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    /* Backdrop */
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">

        {/* Header */}
        <div className="flex justify-between items-center px-6 py-4 border-b bg-gray-50">
          <h2 className="text-lg font-semibold text-gray-800">➕ Add Camera</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none">✕</button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
              {error}
            </div>
          )}

          <Field label="Camera ID *" hint="Unique identifier, e.g. entrance-cam">
            <input
              name="camera_id"
              value={form.camera_id}
              onChange={handleChange}
              placeholder="entrance-cam"
              className="input"
            />
          </Field>

          <Field label="Camera Name" hint="Display name (optional)">
            <input
              name="camera_name"
              value={form.camera_name}
              onChange={handleChange}
              placeholder="Main Entrance"
              className="input"
            />
          </Field>

          <Field label="RTSP URL *" hint="rtsp://user:pass@ip:554/stream or file path">
            <input
              name="rtsp_url"
              value={form.rtsp_url}
              onChange={handleChange}
              placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
              className="input font-mono text-xs"
            />
          </Field>

          <Field label={`Frame Skip: ${form.frame_skip}`} hint="Process every Nth frame (higher = less CPU)">
            <input
              type="range"
              name="frame_skip"
              min={1} max={10}
              value={form.frame_skip}
              onChange={handleChange}
              className="w-full accent-blue-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-0.5">
              <span>1 (every frame)</span>
              <span>10 (low CPU)</span>
            </div>
          </Field>

          {/* Quick presets */}
          <div>
            <p className="text-xs text-gray-400 mb-1 font-medium">Quick URL formats:</p>
            <div className="space-y-1 text-xs text-gray-500 font-mono bg-gray-50 rounded p-2">
              <p>rtsp://admin:pass@<span className="text-blue-500">IP</span>:554/stream1</p>
              <p>rtsp://admin:pass@<span className="text-blue-500">IP</span>:554/Streaming/Channels/101</p>
              <p>/app/storage/<span className="text-blue-500">video.mp4</span> <span className="font-sans text-gray-400">(local file)</span></p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-100"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="px-5 py-2 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 font-medium"
          >
            {loading ? 'Adding…' : 'Add & Start'}
          </button>
        </div>
      </div>

      <style>{`
        .input {
          width: 100%;
          border: 1px solid #d1d5db;
          border-radius: 0.5rem;
          padding: 0.5rem 0.75rem;
          font-size: 0.875rem;
          outline: none;
          transition: border-color 0.15s;
        }
        .input:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px #eff6ff; }
      `}</style>
    </div>
  );
}

function Field({ label, hint, children }) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      {children}
      {hint && <p className="text-xs text-gray-400 mt-0.5">{hint}</p>}
    </div>
  );
}