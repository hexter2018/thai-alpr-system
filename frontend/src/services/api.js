import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ALPR Services
export const alprService = {
  // Get pending verifications
  getPendingVerifications: (page = 1, limit = 20) => 
    api.get('/api/alpr/pending', { params: { page, limit } }),

  // Get all detections
  getDetections: (page = 1, limit = 20, filters = {}) =>
    api.get('/api/alpr/detections', { params: { page, limit, ...filters } }),

  // Get single detection
  getDetection: (id) => 
    api.get(`/api/alpr/detections/${id}`),

  // Verify detection
  verifyDetection: (id, data) =>
    api.post(`/api/alpr/verify/${id}`, data),

  // Process image upload
  processImage: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/api/alpr/process-image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  // Batch process
  batchProcess: (files) => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    return api.post('/api/alpr/batch-process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};

// Vehicle Services
export const vehicleService = {
  // List vehicles
  getVehicles: (page = 1, limit = 20, search = '') =>
    api.get('/api/vehicles', { params: { page, limit, search } }),

  // Get vehicle by ID
  getVehicle: (id) => 
    api.get(`/api/vehicles/${id}`),

  // Create vehicle
  createVehicle: (data) => 
    api.post('/api/vehicles', data),

  // Update vehicle
  updateVehicle: (id, data) =>
    api.put(`/api/vehicles/${id}`, data),

  // Delete vehicle
  deleteVehicle: (id) => 
    api.delete(`/api/vehicles/${id}`),

  // Search by plate
  searchByPlate: (plate) =>
    api.get('/api/vehicles/search', { params: { plate } }),
};

// Statistics Services
export const statsService = {
  // Dashboard KPIs
  getDashboardStats: () => 
    api.get('/api/stats/dashboard'),

  // Date range stats
  getDateRangeStats: (startDate, endDate) =>
    api.get('/api/stats/range', { 
      params: { 
        start_date: startDate, 
        end_date: endDate 
      } 
    }),

  // Camera stats
  getCameraStats: (cameraId) =>
    api.get(`/api/stats/camera/${cameraId}`),
};

// Camera Services
export const cameraService = {
  // List cameras
  getCameras: () => 
    api.get('/api/cameras'),

  // Get camera config
  getCamera: (id) => 
    api.get(`/api/cameras/${id}`),

  // Create camera
  createCamera: (data) => 
    api.post('/api/cameras', data),

  // Update camera
  updateCamera: (id, data) =>
    api.put(`/api/cameras/${id}`, data),

  // Delete camera
  deleteCamera: (id) => 
    api.delete(`/api/cameras/${id}`),

  // Start stream processing
  startStream: (cameraId) =>
    api.post(`/api/stream/start/${cameraId}`),

  // Stop stream processing
  stopStream: (cameraId) =>
    api.post(`/api/stream/stop/${cameraId}`),
};

// System Services
export const systemService = {
  // Health check
  healthCheck: () => 
    api.get('/health'),

  // Get system info
  getSystemInfo: () => 
    api.get('/api/system/info'),
};

export default api;