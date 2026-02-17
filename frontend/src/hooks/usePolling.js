import { useState, useEffect } from 'react';
export function usePolling(fn, interval = 5000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const fetch = async () => {
      try { setData(await fn()); } catch (e) { console.error(e); } finally { setLoading(false); }
    };
    fetch();
    const id = setInterval(fetch, interval);
    return () => clearInterval(id);
  }, [fn, interval]);
  return { data, loading };
}