import React, { useState } from 'react';

export default function PlateEditor({ data, onVerify, onCancel }) {
  const [plate, setPlate] = useState(data.detected_plate);
  const [province, setProvince] = useState(data.detected_province || '');
  const [notes, setNotes] = useState('');
  
  const handleApprove = () => onVerify(data.id, { corrected_plate: plate, corrected_province: province, status: 'ALPR_AUTO', verified_by: 'operator', verification_notes: notes });
  const handleCorrect = () => onVerify(data.id, { corrected_plate: plate, corrected_province: province, status: 'MLPR', verified_by: 'operator', verification_notes: notes });
  const handleReject = () => onVerify(data.id, { corrected_plate: plate, corrected_province: province, status: 'REJECTED', verified_by: 'operator', verification_notes: notes || 'Rejected' });
  
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold mb-4">Edit Detection</h2>
      <div className="space-y-4">
        <div className="w-full h-40 bg-gray-200 rounded flex items-center justify-center text-gray-500">Plate Image</div>
        <div><label className="block text-sm font-medium mb-2">License Plate</label><input type="text" value={plate} onChange={(e) => setPlate(e.target.value)} className="w-full px-4 py-2 border rounded focus:ring-2 focus:ring-blue-500" /></div>
        <div><label className="block text-sm font-medium mb-2">Province</label><input type="text" value={province} onChange={(e) => setProvince(e.target.value)} className="w-full px-4 py-2 border rounded focus:ring-2 focus:ring-blue-500" /></div>
        <div><label className="block text-sm font-medium mb-2">Notes</label><textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows="2" className="w-full px-4 py-2 border rounded focus:ring-2 focus:ring-blue-500"></textarea></div>
        <div className="flex space-x-2">
          <button onClick={handleApprove} className="flex-1 bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">✓ Approve</button>
          <button onClick={handleCorrect} className="flex-1 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">✏️ Correct</button>
          <button onClick={handleReject} className="flex-1 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700">✕ Reject</button>
        </div>
        <button onClick={onCancel} className="w-full bg-gray-200 px-4 py-2 rounded hover:bg-gray-300">Cancel</button>
      </div>
    </div>
  );
}