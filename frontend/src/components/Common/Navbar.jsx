import React from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();
  
  const isActive = (path) => location.pathname === path;
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/live', label: 'Live Monitor', icon: '📹' },
    { path: '/verification', label: 'Verification', icon: '✓' },
    { path: '/master-data', label: 'Master Data', icon: '🚗' },
  ];
  
  return (
    <nav className="bg-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">🇹🇭</span>
            <span className="text-xl font-bold text-gray-800">Thai ALPR System</span>
          </div>
          
          <div className="flex space-x-1">
            {navItems.map(({ path, label, icon }) => (
              <Link
                key={path}
                to={path}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(path)
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <span className="mr-2">{icon}</span>
                {label}
              </Link>
            ))}
          </div>
          
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1 bg-green-100 rounded-full">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              <span className="text-sm font-medium text-green-800">Online</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}