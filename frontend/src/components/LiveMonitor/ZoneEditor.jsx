import React, { useRef, useState, useEffect, useCallback } from 'react';

/**
 * ZoneEditor — draw / edit a polygon detection zone on a canvas.
 *
 * Since browsers can't render an RTSP stream directly, the canvas shows
 * a dark "camera background" overlay.  Users can:
 *  - Click to add points
 *  - Drag existing points to move them
 *  - Double-click a point to delete it
 *  - Press "Clear" to remove all points
 *  - Press "Save Zone" to persist (minimum 3 points)
 */
export default function ZoneEditor({
  cameraId,
  cameraName,
  initialZone,
  onSave,
  saving = false,
  saved  = false,
}) {
  const canvasRef = useRef(null);

  // Points: [{x, y}] — relative to canvas size (0–canvasW / 0–canvasH)
  const [points, setPoints]       = useState([]);
  const [dragging, setDragging]   = useState(null); // index of dragged point
  const [hovered, setHovered]     = useState(null); // index of hovered point
  const [canvasSize, setCanvasSize] = useState({ w: 960, h: 540 });

  const POINT_R    = 7;   // px radius of handle circles
  const SNAP_DIST  = 14;  // px for hit detection

  // ── Initialise from prop ──────────────────────────────────────────────────
  useEffect(() => {
    if (initialZone && initialZone.length > 0) {
      setPoints(
        initialZone
          .map((p) => {
            if (Array.isArray(p) && p.length >= 2) {
              return { x: Number(p[0]), y: Number(p[1]) };
            }
            if (p && typeof p === 'object' && 'x' in p && 'y' in p) {
              return { x: Number(p.x), y: Number(p.y) };
            }
            return null;
          })
          .filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y)),
      );
    } else {
      setPoints([]);
    }
  }, [initialZone]);

  // ── Resize observer ───────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setCanvasSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(canvas.parentElement);
    return () => ro.disconnect();
  }, []);

  // ── Draw ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width  = canvasSize.w;
    canvas.height = canvasSize.h;
    const ctx = canvas.getContext('2d');

    // Background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    ctx.strokeStyle = 'rgba(148,163,184,0.07)';
    ctx.lineWidth   = 1;
    const step = 40;
    for (let x = 0; x < canvas.width; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

    // Camera label
    ctx.font         = '13px monospace';
    ctx.fillStyle    = 'rgba(148,163,184,0.35)';
    ctx.textAlign    = 'center';
    ctx.fillText(`[ ${cameraName || cameraId} ]`, canvas.width / 2, canvas.height / 2);
    ctx.fillText('Click to add zone points', canvas.width / 2, canvas.height / 2 + 22);

    if (points.length === 0) return;

    // Filled polygon
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.fillStyle   = 'rgba(56,189,248,0.12)';
    ctx.fill();

    // Stroke
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth   = 2;
    ctx.setLineDash([6, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Closing line preview (thin)
    if (points.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(points[points.length - 1].x, points[points.length - 1].y);
      ctx.lineTo(points[0].x, points[0].y);
      ctx.strokeStyle = 'rgba(56,189,248,0.35)';
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Points
    points.forEach((p, i) => {
      const isHov = i === hovered;
      const isDrag = i === dragging;

      // Outer glow
      if (isHov || isDrag) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, POINT_R + 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(56,189,248,0.20)';
        ctx.fill();
      }

      // Circle
      ctx.beginPath();
      ctx.arc(p.x, p.y, POINT_R, 0, Math.PI * 2);
      ctx.fillStyle = isHov || isDrag ? '#38bdf8' : '#0f172a';
      ctx.fill();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth   = 2;
      ctx.stroke();

      // Index label
      ctx.font      = 'bold 10px monospace';
      ctx.fillStyle = isHov || isDrag ? '#0f172a' : '#38bdf8';
      ctx.textAlign = 'center';
      ctx.fillText(i + 1, p.x, p.y + 4);
    });

    // Point count
    ctx.font      = '11px monospace';
    ctx.fillStyle = 'rgba(148,163,184,0.6)';
    ctx.textAlign = 'left';
    ctx.fillText(`${points.length} point${points.length !== 1 ? 's' : ''}`, 10, 20);
    if (points.length >= 3) {
      ctx.fillStyle = '#4ade80';
      ctx.fillText('✓ valid zone', 10, 36);
    } else if (points.length > 0) {
      ctx.fillStyle = '#fbbf24';
      ctx.fillText(`need ${3 - points.length} more`, 10, 36);
    }

  }, [points, hovered, dragging, canvasSize, cameraId, cameraName]);

  // ── Event helpers ─────────────────────────────────────────────────────────
  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const nearestPoint = (pos) => {
    let best = null, bestD = Infinity;
    points.forEach((p, i) => {
      const d = Math.hypot(p.x - pos.x, p.y - pos.y);
      if (d < bestD) { bestD = d; best = i; }
    });
    return bestD < SNAP_DIST ? best : null;
  };

  // ── Mouse events ──────────────────────────────────────────────────────────
  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    const pos  = getPos(e);
    const near = nearestPoint(pos);
    if (near !== null) {
      setDragging(near);
    } else {
      setPoints(prev => [...prev, pos]);
    }
  }, [points]);

  const onMouseMove = useCallback((e) => {
    const pos = getPos(e);
    if (dragging !== null) {
      setPoints(prev => prev.map((p, i) => i === dragging ? pos : p));
    } else {
      setHovered(nearestPoint(pos));
    }
  }, [dragging, points]);

  const onMouseUp = useCallback(() => setDragging(null), []);

  const onDoubleClick = useCallback((e) => {
    const pos  = getPos(e);
    const near = nearestPoint(pos);
    if (near !== null) {
      setPoints(prev => prev.filter((_, i) => i !== near));
    }
  }, [points]);

  // ── Touch events ──────────────────────────────────────────────────────────
  const onTouchStart = useCallback((e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect  = canvasRef.current.getBoundingClientRect();
    const pos   = { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
    const near  = nearestPoint(pos);
    if (near !== null) setDragging(near);
    else setPoints(prev => [...prev, pos]);
  }, [points]);

  const onTouchMove = useCallback((e) => {
    e.preventDefault();
    if (dragging === null) return;
    const touch = e.touches[0];
    const rect  = canvasRef.current.getBoundingClientRect();
    const pos   = { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
    setPoints(prev => prev.map((p, i) => i === dragging ? pos : p));
  }, [dragging]);

  const onTouchEnd = useCallback(() => setDragging(null), []);

  // ── Helpers ───────────────────────────────────────────────────────────────
  const handleClear = () => setPoints([]);

  const handleSave = () => {
    if (points.length > 0 && points.length < 3) {
      return; // don't save invalid zone
    }
    onSave(points);
  };

  const canSave = points.length === 0 || points.length >= 3;

  return (
    <div className="space-y-4">

      {/* Instructions */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 text-xs text-gray-400 space-y-1">
        <p className="text-gray-200 font-medium mb-1">Zone Editor — {cameraName || cameraId}</p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-0.5">
          <span>🖱 Click canvas → add point</span>
          <span>🖱 Drag point → move</span>
          <span>🖱 Double-click point → delete</span>
          <span>📐 Min 3 points for valid zone</span>
        </div>
      </div>

      {/* Canvas */}
      <div className="relative w-full rounded-xl overflow-hidden border border-gray-800" style={{ aspectRatio: '16/9' }}>
        <canvas
          ref={canvasRef}
          className="w-full h-full cursor-crosshair select-none"
          style={{ touchAction: 'none' }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
          onDoubleClick={onDoubleClick}
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
          onTouchEnd={onTouchEnd}
        />
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            onClick={handleClear}
            disabled={points.length === 0}
            className="px-4 py-2 text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-gray-300 transition disabled:opacity-40"
          >
            ✕ Clear Zone
          </button>

          {points.length > 0 && points.length < 3 && (
            <span className="text-xs text-amber-400">
              ⚠ Need at least 3 points ({3 - points.length} more)
            </span>
          )}

          {points.length >= 3 && (
            <span className="text-xs text-emerald-400">
              ✓ {points.length} points — zone ready
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {saved && (
            <span className="text-xs text-emerald-400 animate-pulse">✓ Saved!</span>
          )}
          <button
            onClick={handleSave}
            disabled={!canSave || saving}
            className="px-5 py-2 text-xs font-medium bg-sky-600 hover:bg-sky-500 disabled:opacity-40 rounded text-white transition"
          >
            {saving ? '⟳ Saving…' : '💾 Save Zone'}
          </button>
        </div>
      </div>

      {/* Current points list */}
      {points.length > 0 && (
        <details className="group">
          <summary className="text-xs text-gray-600 cursor-pointer hover:text-gray-400 select-none">
            ▶ Show raw coordinates ({points.length} points)
          </summary>
          <div className="mt-2 bg-gray-900 rounded border border-gray-800 p-3 max-h-40 overflow-auto">
            <pre className="text-[10px] text-gray-500 font-mono">
              {JSON.stringify(points.map((p, i) => ({
                index: i + 1,
                x: Math.round(p.x),
                y: Math.round(p.y),
              })), null, 2)}
            </pre>
          </div>
        </details>
      )}
    </div>
  );
}