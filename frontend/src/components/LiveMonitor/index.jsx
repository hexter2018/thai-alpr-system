import React, { useState, useEffect } from 'react';
import StreamViewer from './StreamViewer';
import DetectionOverlay from './DetectionOverlay';
import LogTable from './LogTable';
import { useWebSocket } from '../../hooks/useWebSocket';

export default function LiveMonitor() {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const { messages, isConnected } = useWebSocket(selectedCamera);
  
  useEffect(() => {
    loadCameras();
  }, []);
  
  const loadCameras = async () => {
    try {
      const response = await fetch('/api/stream/list');
      const data = await response.json();
      setCameras(Object.keys(data.cameras));
      if (Object.keys(data.cameras).length > 0) setSelectedCamera(Object.keys(data.cameras)[0]);
    } catch (error) {
      console.error('Failed to load cameras:', error);
    }
  };
  
  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div><h1 className="text-3xl font-bold">Live Monitor</h1><p className="text-gray-600">Real-time ALPR detection</p></div>
        <div className="flex items-center space-x-2">
          <span className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
          <span className="text-sm font-medium">{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>
      
      <div className="flex space-x-4 mb-4">
        {cameras.map(cam => (
          <button key={cam} onClick={() => setSelectedCamera(cam)} className={`px-4 py-2 rounded ${selectedCamera === cam ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}>{cam}</button>
        ))}
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <StreamViewer cameraId={selectedCamera} />
        <DetectionOverlay detections={messages} />
      </div>
      
      <LogTable logs={messages} />
    </div>
  );
}