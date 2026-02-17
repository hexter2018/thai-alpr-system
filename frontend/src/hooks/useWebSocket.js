import { useState, useEffect, useRef } from 'react';
export function useWebSocket(cameraId) {
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const ws = useRef(null);
  useEffect(() => {
    if (!cameraId) return;
    ws.current = new WebSocket(`${import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}/ws/${cameraId}`);
    ws.current.onopen = () => setIsConnected(true);
    ws.current.onmessage = (e) => { try { setMessages(p => [...p, JSON.parse(e.data)]); } catch(err){} };
    ws.current.onerror = () => setIsConnected(false);
    ws.current.onclose = () => setIsConnected(false);
    return () => ws.current?.close();
  }, [cameraId]);
  return { messages, isConnected };
}