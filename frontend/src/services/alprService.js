import api from './api';
export const alprService = {
  processImage: async (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return (await api.post('/api/alpr/process-image', fd)).data;
  },
  getPendingVerifications: async (skip = 0, limit = 20) => (await api.get(`/api/alpr/pending?skip=${skip}&limit=${limit}`)).data,
  verifyDetection: async (id, data) => (await api.post(`/api/alpr/verify/${id}`, data)).data,
  getVehicles: async (skip = 0, limit = 50) => (await api.get(`/api/vehicles?skip=${skip}&limit=${limit}`)).data,
  createVehicle: async (data) => (await api.post('/api/vehicles', data)).data,
  updateVehicle: async (id, data) => (await api.put(`/api/vehicles/${id}`, data)).data,
  deleteVehicle: async (id) => (await api.delete(`/api/vehicles/${id}`)).data,
  getDashboardStats: async () => (await api.get('/api/stats/dashboard')).data,
  getDailyStats: async (days = 7) => (await api.get(`/api/stats/daily?days=${days}`)).data,
};