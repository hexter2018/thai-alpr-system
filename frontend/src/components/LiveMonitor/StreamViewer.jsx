import React from 'react';

export default function StreamViewer({ cameraId, cameraName, isRunning }) {
  const label = cameraName || cameraId || 'No Camera Selected';

  return (
    <div className="bg-gray-900 rounded-xl overflow-hidden aspect-video relative flex items-center justify-center">
      {isRunning ? (
        <div className="text-center">
          <div className="text-6xl mb-4">📹</div>
          <p className="text-white font-semibold text-lg">{label}</p>
          <p className="text-green-400 text-sm mt-1">● Live</p>
          <p className="text-gray-500 text-xs mt-3 max-w-xs">
            RTSP stream is active — browsers cannot display RTSP directly.
            Detections appear in the panel →
          </p>
        </div>
      ) : (
        <div className="text-center opacity-60">
          <div className="text-6xl mb-4">📷</div>
          <p className="text-gray-400 font-medium">{label}</p>
          <p className="text-gray-600 text-sm mt-1">Camera stopped</p>
          <p className="text-gray-600 text-xs mt-2">Press ▶ above to start</p>
        </div>
      )}

      {/* REC badge */}
      {isRunning && (
        <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/60 px-2 py-1 rounded-full">
          <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span className="text-white text-xs font-semibold">REC</span>
        </div>
      )}

      {/* Camera label badge */}
      <div className="absolute bottom-3 left-3 bg-black/50 text-white text-xs px-2 py-1 rounded">
        {cameraId || '—'}
      </div>
    </div>
  );
}