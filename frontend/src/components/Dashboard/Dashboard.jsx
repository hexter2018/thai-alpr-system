import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { statsService } from '../../services/api';
import {
  ChartBarIcon,
  CheckCircleIcon,
  ClockIcon,
  XCircleIcon,
  TruckIcon,
} from '@heroicons/react/24/outline';
import StatsChart from './StatsChart';

const KPICard = ({ title, value, subtitle, icon: Icon, color, trend }) => {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
          {subtitle && (
            <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
          )}
          {trend && (
            <div className={`text-sm mt-2 ${trend.positive ? 'text-green-600' : 'text-red-600'}`}>
              {trend.positive ? '↑' : '↓'} {trend.value}%
            </div>
          )}
        </div>
        <div className={`${colorClasses[color]} p-3 rounded-lg`}>
          <Icon className="w-8 h-8 text-white" />
        </div>
      </div>
    </div>
  );
};

const Dashboard = () => {
  const [timeRange, setTimeRange] = useState('today');

  // Fetch dashboard stats
  const { data: stats, isLoading, error } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: () => statsService.getDashboardStats(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  const kpis = stats?.data || {
    total_detections: 0,
    alpr_auto_count: 0,
    mlpr_count: 0,
    pending_count: 0,
    rejected_count: 0,
    accuracy_percentage: 0,
    avg_confidence: 0,
    unique_vehicles: 0,
    avg_processing_time_ms: 0,
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">ภาพรวมระบบ ALPR แบบเรียลไทม์</p>
        </div>
        
        {/* Time Range Selector */}
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="input w-48"
        >
          <option value="today">วันนี้</option>
          <option value="week">7 วันที่ผ่านมา</option>
          <option value="month">30 วันที่ผ่านมา</option>
          <option value="all">ทั้งหมด</option>
        </select>
      </div>

      {isLoading && (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">กำลังโหลดข้อมูล...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
          เกิดข้อผิดพลาดในการโหลดข้อมูล
        </div>
      )}

      {!isLoading && !error && (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <KPICard
              title="การตรวจจับทั้งหมด"
              value={kpis.total_detections.toLocaleString()}
              subtitle={`${kpis.unique_vehicles} คันไม่ซ้ำ`}
              icon={TruckIcon}
              color="blue"
            />
            
            <KPICard
              title="ALPR อัตโนมัติ"
              value={kpis.alpr_auto_count.toLocaleString()}
              subtitle={`${((kpis.alpr_auto_count / kpis.total_detections) * 100 || 0).toFixed(1)}%`}
              icon={CheckCircleIcon}
              color="green"
            />
            
            <KPICard
              title="รอตรวจสอบ"
              value={kpis.pending_count.toLocaleString()}
              subtitle="ความเชื่อมั่นต่ำ"
              icon={ClockIcon}
              color="yellow"
            />
            
            <KPICard
              title="แก้ไขด้วยมือ (MLPR)"
              value={kpis.mlpr_count.toLocaleString()}
              subtitle={`${((kpis.mlpr_count / kpis.total_detections) * 100 || 0).toFixed(1)}%`}
              icon={ChartBarIcon}
              color="purple"
            />
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">ความแม่นยำ</h3>
              <div className="flex items-end space-x-2">
                <span className="text-4xl font-bold text-green-600">
                  {kpis.accuracy_percentage.toFixed(1)}%
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                อัตราการตรวจจับที่ถูกต้อง
              </p>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">ความเชื่อมั่นเฉลี่ย</h3>
              <div className="flex items-end space-x-2">
                <span className="text-4xl font-bold text-blue-600">
                  {(kpis.avg_confidence * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                OCR confidence score
              </p>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">เวลาประมวลผล</h3>
              <div className="flex items-end space-x-2">
                <span className="text-4xl font-bold text-purple-600">
                  {kpis.avg_processing_time_ms.toFixed(0)}
                </span>
                <span className="text-xl text-gray-600 mb-1">ms</span>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                เฉลี่ยต่อการตรวจจับ
              </p>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <StatsChart type="detections" timeRange={timeRange} />
            <StatsChart type="accuracy" timeRange={timeRange} />
          </div>

          {/* Status Breakdown */}
          <div className="card">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">สถานะการตรวจจับ</h3>
            <div className="space-y-3">
              <StatusBar
                label="ALPR อัตโนมัติ (>95%)"
                count={kpis.alpr_auto_count}
                total={kpis.total_detections}
                color="bg-green-500"
              />
              <StatusBar
                label="รอตรวจสอบ (≤95%)"
                count={kpis.pending_count}
                total={kpis.total_detections}
                color="bg-yellow-500"
              />
              <StatusBar
                label="แก้ไขแล้ว (MLPR)"
                count={kpis.mlpr_count}
                total={kpis.total_detections}
                color="bg-blue-500"
              />
              <StatusBar
                label="ปฏิเสธ"
                count={kpis.rejected_count}
                total={kpis.total_detections}
                color="bg-red-500"
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
};

const StatusBar = ({ label, count, total, color }) => {
  const percentage = total > 0 ? (count / total) * 100 : 0;

  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        <span className="text-sm text-gray-600">
          {count.toLocaleString()} ({percentage.toFixed(1)}%)
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`${color} h-2 rounded-full transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export default Dashboard;