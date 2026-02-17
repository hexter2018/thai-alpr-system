import React from 'react';

export default function DetectionOverlay({ detections = [] }) {
  // Show last 5, newest first
  const recent = [...detections].slice(-5).reverse();

  return (
    <div className="bg-white rounded-lg shadow h-full flex flex-col">
      <div className="flex justify-between items-center px-4 py-3 border-b">
        <h3 className="text-base font-semibold text-gray-800">Recent Detections</h3>
        <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">
          {detections.length} total
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {recent.length === 0 ? (
          <p className="text-gray-400 text-sm text-center py-10">
            Waiting for detections…
          </p>
        ) : (
          recent.map((det, idx) => {
            // backend sends ocr_confidence; hook normalises to confidence too
            const conf   = det.confidence ?? det.ocr_confidence ?? 0;
            const isHigh = conf >= 0.95;

            return (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-100"
              >
                <div className="min-w-0">
                  <p className="font-bold text-gray-900 truncate">
                    {det.detected_plate || '—'}
                  </p>
                  <p className="text-xs text-gray-500">{det.detected_province || ''}</p>
                  <p className="text-xs text-gray-400">{det.camera_id}</p>
                </div>
                <div className="text-right ml-2 shrink-0">
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-medium ${
                      isHigh
                        ? 'bg-green-100 text-green-700'
                        : 'bg-yellow-100 text-yellow-700'
                    }`}
                  >
                    {(conf * 100).toFixed(0)}%
                  </span>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {det.status || ''}
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}