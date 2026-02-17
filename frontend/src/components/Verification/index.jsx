import React, { useState, useEffect } from 'react';
import VerificationQueue from './VerificationQueue';
import PlateEditor from './PlateEditor';
import { alprService } from '../../services/alprService';

export default function Verification() {
  const [queue, setQueue] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    loadQueue();
  }, []);
  
  const loadQueue = async () => {
    try {
      const data = await alprService.getPendingVerifications();
      setQueue(data);
    } catch (error) {
      console.error('Failed to load queue:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const handleVerify = async (logId, data) => {
    try {
      await alprService.verifyDetection(logId, data);
      await loadQueue();
      setSelected(null);
    } catch (error) {
      console.error('Failed to verify:', error);
    }
  };
  
  if (loading) return <div className="flex items-center justify-center h-screen"><div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600"></div></div>;
  
  return (
    <div className="p-6 space-y-6">
      <div><h1 className="text-3xl font-bold">Verification Queue</h1><p className="text-gray-600">{queue.length} plates pending verification</p></div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <VerificationQueue queue={queue} onSelect={setSelected} selected={selected} />
        {selected && <PlateEditor data={selected} onVerify={handleVerify} onCancel={() => setSelected(null)} />}
      </div>
    </div>
  );
}