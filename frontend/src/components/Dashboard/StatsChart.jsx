import React, { useEffect, useState } from 'react';
import { alprService } from '../../services/alprService';

export default function StatsChart() {
  const [data, setData] = useState([]);
  
  useEffect(() => {
    loadChartData();
  }, []);
  
  const loadChartData = async () => {
    try {
      const result = await alprService.getDailyStats(7);
      setData(result);
    } catch (error) {
      console.error('Failed to load chart data:', error);
    }
  };
  
  const maxValue = Math.max(...data.map(d => d.count), 1);
  
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Daily Detections (Last 7 Days)</h2>
      <div className="flex items-end justify-between h-48 space-x-2">
        {data.map((item, index) => (
          <div key={index} className="flex-1 flex flex-col items-center">
            <div className="w-full bg-blue-500 rounded-t hover:bg-blue-600 transition-colors" style={{ height: `${(item.count / maxValue) * 100}%` }}></div>
            <span className="text-xs text-gray-600 mt-2">{new Date(item.date).toLocaleDateString('th-TH', { month: 'short', day: 'numeric' })}</span>
            <span className="text-xs font-medium text-gray-900">{item.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}