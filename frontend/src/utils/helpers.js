export const formatDate = (d) => new Date(d).toLocaleString('th-TH');
export const formatConfidence = (c) => `${(c * 100).toFixed(0)}%`;
export const getStatusColor = (s) => ({ ALPR_AUTO: 'bg-green-100 text-green-800', PENDING_VERIFY: 'bg-yellow-100 text-yellow-800', MLPR: 'bg-blue-100 text-blue-800', REJECTED: 'bg-red-100 text-red-800' })[s] || 'bg-gray-100 text-gray-800';