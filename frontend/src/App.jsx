import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Common/Navbar';
import Dashboard from './components/Dashboard';
import LiveMonitor from './components/LiveMonitor';
import Verification from './components/Verification';
import MasterData from './components/MasterData';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/live" element={<LiveMonitor />} />
          <Route path="/verification" element={<Verification />} />
          <Route path="/master-data" element={<MasterData />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}