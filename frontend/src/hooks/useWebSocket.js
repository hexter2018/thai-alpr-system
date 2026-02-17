import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * useWebSocket - connects to /ws/{cameraId} and parses messages.
 *
 * Backend sends:
 *   { type: "detection", camera_id, detections: [...], timestamp }
 *   { type: "status",    camera_id, status: "running"|"stopped", stats }
 *   { type: "stats",     camera_id, data: {...} }
 */
export function useWebSocket(cameraId, options = {}) {
  const { reconnectInterval = 3000, maxReconnects = 10 } = options;

  const [messages, setMessages]       = useState([]);   // flattened detection items
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus]           = useState('disconnected');
  const [cameraStats, setCameraStats] = useState(null);

  const wsRef             = useRef(null);
  const reconnectCount    = useRef(0);
  const reconnectTimer    = useRef(null);
  const intentionalClose  = useRef(false);
  const heartbeatInterval = useRef(null);

  const clearTimers = useCallback(() => {
    if (reconnectTimer.current)    clearTimeout(reconnectTimer.current);
    if (heartbeatInterval.current) clearInterval(heartbeatInterval.current);
  }, []);

  const connect = useCallback(() => {
    if (!cameraId) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const proto  = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl  = `${proto}://${window.location.host}/ws/${cameraId}`;

    setStatus('connecting');

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setStatus('connected');
        reconnectCount.current = 0;

        // Heartbeat every 30 s
        heartbeatInterval.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping');
        }, 30_000);
      };

      ws.onmessage = (event) => {
        if (event.data === 'pong') return;

        try {
          const msg = JSON.parse(event.data);

          if (msg.type === 'detection') {
            // Normalise each detection so the UI always has consistent fields
            const items = (msg.detections || []).map((det) => ({
              camera_id:         msg.camera_id,
              detected_plate:    det.detected_plate   || '',
              detected_province: det.detected_province || '',
              confidence:        det.ocr_confidence   ?? det.confidence ?? 0,
              ocr_confidence:    det.ocr_confidence   ?? det.confidence ?? 0,
              status:            det.status           || '',
              vehicle_type:      det.vehicle_type     || '',
              timestamp:         det.timestamp        || msg.timestamp || new Date().toISOString(),
              tracking_id:       det.tracking_id      || '',
            }));

            // Keep last 200 entries
            setMessages((prev) => [...prev.slice(-200), ...items]);

          } else if (msg.type === 'status') {
            setStatus(msg.status || 'unknown');
            if (msg.stats) setCameraStats(msg.stats);

          } else if (msg.type === 'stats') {
            setCameraStats(msg.data);
          }

        } catch (_) { /* ignore non-JSON */ }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setStatus('disconnected');
        clearInterval(heartbeatInterval.current);

        if (!intentionalClose.current && reconnectCount.current < maxReconnects) {
          reconnectCount.current += 1;
          setStatus('reconnecting');
          reconnectTimer.current = setTimeout(connect, reconnectInterval);
        }
      };

      ws.onerror = () => setStatus('error');

    } catch (err) {
      console.error('WebSocket connect error:', err);
      setStatus('error');
    }
  }, [cameraId, reconnectInterval, maxReconnects, clearTimers]);

  const disconnect = useCallback(() => {
    intentionalClose.current = true;
    clearTimers();
    wsRef.current?.close();
    wsRef.current = null;
    setIsConnected(false);
    setStatus('disconnected');
  }, [clearTimers]);

  const sendMessage = useCallback((msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
  }, []);

  const clearMessages = useCallback(() => setMessages([]), []);

  // (Re)connect whenever cameraId changes
  useEffect(() => {
    if (!cameraId) return;

    intentionalClose.current = false;
    setMessages([]);
    setCameraStats(null);
    connect();

    return () => {
      intentionalClose.current = true;
      clearTimers();
      wsRef.current?.close();
    };
  }, [cameraId]); // eslint-disable-line react-hooks/exhaustive-deps

  return { messages, isConnected, status, cameraStats, sendMessage, clearMessages, disconnect };
}