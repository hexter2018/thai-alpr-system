import React from 'react';

export default function DetectionOverlay({ detections = [] }) {
  const recent = detections.slice(-5);
  
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Recent Detections</h3>
      <div className="space-y-3">
        {recent.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No detections yet</p>
        ) : (
          recent.map((det, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-gray-300 rounded"></div>
                <div>
                  <p className="font-medium">{det.detected_plate}</p>
                  <p className="text-sm text-gray-600">{det.detected_province}</p>
                </div>
              </div>
              <div className="text-right">
                <span className={`text-xs px-2 py-1 rounded ${det.confidence > 0.95 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                  {(det.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}