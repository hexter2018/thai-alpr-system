import React from 'react';

export default function LogTable({ logs = [] }) {
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Plate</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Province</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {logs.slice(-10).reverse().map((log, idx) => (
            <tr key={idx} className="hover:bg-gray-50">
              <td className="px-6 py-4 whitespace-nowrap text-sm">{new Date(log.timestamp).toLocaleTimeString()}</td>
              <td className="px-6 py-4 whitespace-nowrap font-medium">{log.detected_plate}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm">{log.detected_province}</td>
              <td className="px-6 py-4 whitespace-nowrap"><span className={`px-2 py-1 rounded text-xs ${log.confidence > 0.95 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>{(log.confidence * 100).toFixed(0)}%</span></td>
              <td className="px-6 py-4 whitespace-nowrap text-sm">{log.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}