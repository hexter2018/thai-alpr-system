import React from 'react';

export default function StreamViewer({ cameraId }) {
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden aspect-video">
      <div className="relative w-full h-full flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">📹</div>
          <p className="text-white text-lg font-medium">{cameraId || 'No Camera Selected'}</p>
          <p className="text-gray-400 text-sm mt-2">RTSP Stream Display</p>
        </div>
      </div>
    </div>
  );
}