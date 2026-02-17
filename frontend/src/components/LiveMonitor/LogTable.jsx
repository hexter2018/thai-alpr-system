import React from 'react';

export default function LogTable({ logs = [], onClear }) {
  const rows = [...logs].slice(-50).reverse();

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      {/* Header */}
      <div className="flex justify-between items-center px-5 py-3 border-b bg-gray-50">
        <h3 className="font-semibold text-gray-700 text-sm">Detection Log</h3>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-400">{logs.length} entries</span>
          {onClear && (
            <button
              onClick={onClear}
              className="text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto max-h-60 overflow-y-auto">
        <table className="min-w-full text-sm divide-y divide-gray-100">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr>
              {['Time', 'Camera', 'Plate', 'Province', 'Confidence', 'Status'].map((h) => (
                <th
                  key={h}
                  className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wide"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-50">
            {rows.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-400 text-sm">
                  No detections yet
                </td>
              </tr>
            ) : (
              rows.map((log, idx) => {
                // backend sends ocr_confidence; hook normalises both fields
                const conf   = log.confidence ?? log.ocr_confidence ?? 0;
                const isHigh = conf >= 0.95;

                return (
                  <tr key={idx} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-2 text-xs text-gray-500 whitespace-nowrap">
                      {log.timestamp
                        ? new Date(log.timestamp).toLocaleTimeString('th-TH')
                        : '—'}
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-600 whitespace-nowrap">
                      {log.camera_id || '—'}
                    </td>
                    <td className="px-4 py-2 font-bold text-gray-900 whitespace-nowrap">
                      {log.detected_plate || '—'}
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-600">
                      {log.detected_province || '—'}
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className={`text-xs px-2 py-0.5 rounded font-medium ${
                          isHigh
                            ? 'bg-green-100 text-green-700'
                            : 'bg-yellow-100 text-yellow-700'
                        }`}
                      >
                        {(conf * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-500">
                      {log.status || '—'}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}