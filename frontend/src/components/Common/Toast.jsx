import React, { useEffect } from 'react';

export default function Toast({ message, type = 'info', onClose, duration = 3000 }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, duration);
    
    return () => clearTimeout(timer);
  }, [duration, onClose]);
  
  const bgColors = {
    success: 'bg-green-500',
    error: 'bg-red-500',
    warning: 'bg-yellow-500',
    info: 'bg-blue-500',
  };
  
  const icons = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ',
  };
  
  return (
    <div className={`${bgColors[type]} text-white px-6 py-4 rounded-lg shadow-lg flex items-center space-x-3 animate-slide-in`}>
      <span className="text-xl">{icons[type]}</span>
      <span className="font-medium">{message}</span>
      <button onClick={onClose} className="ml-4 text-white hover:text-gray-200">
        ✕
      </button>
    </div>
  );
}