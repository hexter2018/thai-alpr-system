import React from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function Sidebar() {
  const location = useLocation();
  
  const menuItems = [
    { path: '/', icon: '📊', label: 'Dashboard' },
    { path: '/live', icon: '📹', label: 'Live Monitor' },
    { path: '/verification', icon: '✓', label: 'Verification' },
    { path: '/master-data', icon: '🚗', label: 'Master Data' },
    { path: '/logs', icon: '📋', label: 'Access Logs' },
    { path: '/settings', icon: '⚙️', label: 'Settings' },
  ];
  
  return (
    <aside className="w-64 bg-gray-900 min-h-screen text-white">
      <div className="p-4">
        <div className="flex items-center space-x-2 mb-8">
          <span className="text-3xl">🇹🇭</span>
          <div>
            <h1 className="text-lg font-bold">Thai ALPR</h1>
            <p className="text-xs text-gray-400">System v1.0</p>
          </div>
        </div>
        
        <nav className="space-y-1">
          {menuItems.map(({ path, icon, label }) => {
            const isActive = location.pathname === path;
            return (
              <Link
                key={path}
                to={path}
                className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-800'
                }`}
              >
                <span className="text-xl">{icon}</span>
                <span className="font-medium">{label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}