import React, { useState, useEffect } from 'react';

export default function StreamViewer({ cameraId, cameraName, isRunning }) {
  const label = cameraName || cameraId || 'No Camera Selected';
  const [imageError, setImageError] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  // สร้าง URL สำหรับดึงภาพ MJPEG
  // ใช้ VITE_API_URL จาก .env หรือ fallback เป็น localhost
  const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const streamUrl = isRunning 
    ? `${baseUrl}/api/stream/video/${cameraId}?t=${refreshKey}` 
    : null;

  useEffect(() => {
    // รีเซ็ตสถานะเมื่อเปลี่ยนกล้อง
    setImageError(false);
    setRefreshKey(Date.now());
  }, [cameraId, isRunning]);

  return (
    <div className="bg-gray-900 rounded-xl overflow-hidden aspect-video relative flex items-center justify-center border border-gray-800">
      {isRunning && !imageError ? (
        // --- ส่วนแสดงผลวิดีโอ (ใช้ img tag ธรรมดารับ MJPEG) ---
        <img 
            src={streamUrl}
            alt={`Live ${label}`}
            className="w-full h-full object-contain"
            onError={() => setImageError(true)}
        />
      ) : (
        // --- ส่วน Placeholder (เมื่อหยุด หรือ โหลดภาพไม่ได้) ---
        <div className="text-center opacity-60">
          <div className="text-6xl mb-4">
            {isRunning ? '📡' : '📷'}
          </div>
          <p className="text-gray-400 font-medium text-xl">{label}</p>
          
          {isRunning ? (
            <div className="mt-2">
              <p className="text-yellow-500 text-sm font-semibold animate-pulse">
                Connecting to stream...
              </p>
              <p className="text-gray-600 text-xs mt-1">
                Waiting for backend feed
              </p>
            </div>
          ) : (
             <div className="mt-2">
                <p className="text-gray-600 text-sm">Camera stopped</p>
                <p className="text-gray-600 text-xs mt-1">Press ▶ to start</p>
             </div>
          )}
        </div>
      )}

      {/* Overlay Status Badge */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
         {isRunning && !imageError && (
            <div className="flex items-center gap-2 bg-black/70 backdrop-blur-sm px-3 py-1.5 rounded-full border border-green-500/30 shadow-lg">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                </span>
                <span className="text-green-400 text-xs font-bold tracking-wide">LIVE</span>
            </div>
         )}
         
         {imageError && isRunning && (
            <div className="flex items-center gap-2 bg-black/70 backdrop-blur-sm px-3 py-1.5 rounded-full border border-red-500/30">
                <span className="h-2 w-2 rounded-full bg-red-500"></span>
                <span className="text-red-400 text-xs font-bold">NO SIGNAL</span>
            </div>
         )}
      </div>

      {/* Camera Info Overlay (Bottom Left) */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-4 pt-12 pointer-events-none">
        <h3 className="text-white font-medium text-lg drop-shadow-md">{label}</h3>
        {isRunning && (
             <p className="text-gray-400 text-xs font-mono mt-0.5">ID: {cameraId}</p>
        )}
      </div>
    </div>
  );
}