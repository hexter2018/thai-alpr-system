import React, { useState, useEffect } from 'react';
import VehicleTable from './VehicleTable';
import VehicleForm from './VehicleForm';
import { alprService } from '../../services/alprService';

export default function MasterData() {
  const [vehicles, setVehicles] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [editing, setEditing] = useState(null);
  
  useEffect(() => {
    loadVehicles();
  }, []);
  
  const loadVehicles = async () => {
    try {
      const data = await alprService.getVehicles();
      setVehicles(data);
    } catch (error) {
      console.error('Failed to load vehicles:', error);
    }
  };
  
  const handleSave = async (data) => {
    try {
      if (editing) {
        await alprService.updateVehicle(editing.id, data);
      } else {
        await alprService.createVehicle(data);
      }
      await loadVehicles();
      setShowForm(false);
      setEditing(null);
    } catch (error) {
      console.error('Failed to save vehicle:', error);
    }
  };
  
  const handleEdit = (vehicle) => {
    setEditing(vehicle);
    setShowForm(true);
  };
  
  const handleDelete = async (id) => {
    if (confirm('Delete this vehicle?')) {
      try {
        await alprService.deleteVehicle(id);
        await loadVehicles();
      } catch (error) {
        console.error('Failed to delete vehicle:', error);
      }
    }
  };
  
  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div><h1 className="text-3xl font-bold">Master Data</h1><p className="text-gray-600">Manage registered vehicles</p></div>
        <button onClick={() => { setShowForm(true); setEditing(null); }} className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">+ Add Vehicle</button>
      </div>
      {showForm && <VehicleForm data={editing} onSave={handleSave} onCancel={() => { setShowForm(false); setEditing(null); }} />}
      <VehicleTable vehicles={vehicles} onEdit={handleEdit} onDelete={handleDelete} />
    </div>
  );
}