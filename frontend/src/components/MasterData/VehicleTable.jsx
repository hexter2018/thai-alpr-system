import React from 'react';

export default function VehicleTable({ vehicles, onEdit, onDelete }) {
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Plate</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Province</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Owner</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {vehicles.map(v => (
            <tr key={v.id} className="hover:bg-gray-50">
              <td className="px-6 py-4 whitespace-nowrap font-medium">{v.license_plate}</td>
              <td className="px-6 py-4 whitespace-nowrap">{v.province}</td>
              <td className="px-6 py-4 whitespace-nowrap">{v.vehicle_type}</td>
              <td className="px-6 py-4 whitespace-nowrap">{v.owner_name || '-'}</td>
              <td className="px-6 py-4 whitespace-nowrap"><span className={`px-2 py-1 text-xs rounded ${v.is_authorized ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>{v.is_authorized ? 'Authorized' : 'Blocked'}</span></td>
              <td className="px-6 py-4 whitespace-nowrap space-x-2"><button onClick={() => onEdit(v)} className="text-blue-600 hover:underline">Edit</button><button onClick={() => onDelete(v.id)} className="text-red-600 hover:underline">Delete</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
