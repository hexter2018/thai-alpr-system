import React, { useState, useEffect } from 'react';

export default function VehicleForm({ data, onSave, onCancel }) {
  const [form, setForm] = useState({
    license_plate: '',
    province: '',
    vehicle_type: 'car',
    brand: '',
    model: '',
    color: '',
    owner_name: '',
    owner_phone: '',
    is_authorized: true,
    notes: ''
  });
  
  useEffect(() => {
    if (data) setForm(data);
  }, [data]);
  
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm(prev => ({ ...prev, [name]: type === 'checkbox' ? checked : value }));
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(form);
  };
  
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">{data ? 'Edit Vehicle' : 'Add Vehicle'}</h2>
      <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
        <div><label className="block text-sm font-medium mb-1">License Plate *</label><input name="license_plate" value={form.license_plate} onChange={handleChange} required className="w-full px-3 py-2 border rounded" /></div>
        <div><label className="block text-sm font-medium mb-1">Province *</label><input name="province" value={form.province} onChange={handleChange} required className="w-full px-3 py-2 border rounded" /></div>
        <div><label className="block text-sm font-medium mb-1">Type</label><select name="vehicle_type" value={form.vehicle_type} onChange={handleChange} className="w-full px-3 py-2 border rounded"><option value="car">Car</option><option value="truck">Truck</option><option value="motorcycle">Motorcycle</option><option value="van">Van</option><option value="bus">Bus</option></select></div>
        <div><label className="block text-sm font-medium mb-1">Brand</label><input name="brand" value={form.brand} onChange={handleChange} className="w-full px-3 py-2 border rounded" /></div>
        <div><label className="block text-sm font-medium mb-1">Model</label><input name="model" value={form.model} onChange={handleChange} className="w-full px-3 py-2 border rounded" /></div>
        <div><label className="block text-sm font-medium mb-1">Color</label><input name="color" value={form.color} onChange={handleChange} className="w-full px-3 py-2 border rounded" /></div>
        <div><label className="block text-sm font-medium mb-1">Owner Name</label><input name="owner_name" value={form.owner_name} onChange={handleChange} className="w-full px-3 py-2 border rounded" /></div>
        <div><label className="block text-sm font-medium mb-1">Owner Phone</label><input name="owner_phone" value={form.owner_phone} onChange={handleChange} className="w-full px-3 py-2 border rounded" /></div>
        <div className="col-span-2"><label className="flex items-center"><input type="checkbox" name="is_authorized" checked={form.is_authorized} onChange={handleChange} className="mr-2" /> Authorized</label></div>
        <div className="col-span-2"><label className="block text-sm font-medium mb-1">Notes</label><textarea name="notes" value={form.notes} onChange={handleChange} rows="2" className="w-full px-3 py-2 border rounded"></textarea></div>
        <div className="col-span-2 flex space-x-2"><button type="submit" className="flex-1 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">Save</button><button type="button" onClick={onCancel} className="flex-1 bg-gray-200 px-4 py-2 rounded hover:bg-gray-300">Cancel</button></div>
      </form>
    </div>
  );
}
