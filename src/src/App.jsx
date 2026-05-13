import React, { useState, useRef, useCallback } from 'react';
import './App.css';

const { ipcRenderer } = window.require('electron');

// Windows-safe file:// URL
const toFileUrl = (p) => {
  if (!p) return '';
  const s = p.replace(/\\/g, '/');
  return s.startsWith('/') ? `file://${s}` : `file:///${s}`;
};

const PANELS = [
  { key: 'original',   label: 'Temperature',    icon: '🌡',  desc: 'Raw °C values' },
  { key: 'magnitude',  label: 'Gradient Mag',   icon: '〰',  desc: 'Overall edge strength' },
  { key: 'mag_thresh', label: 'Strong Edges',   icon: '⚡',  desc: 'Top 25% gradient zones' },
  { key: 'angle',      label: 'Flow Angle',     icon: '🧭',  desc: 'Gradient direction (HSV)' },
  { key: 'overlay',    label: 'Overlay',        icon: '🔲',  desc: 'Edges on temperature' },
  { key: 'quiver',     label: 'Quiver Arrows',  icon: '↗',  desc: 'Heat flow arrows' },
  { key: 'grid',       label: 'Full Grid',      icon: '⊞',  desc: 'All panels combined' },
];

const DIR_COLORS = { N: '#7eb8f7', S: '#f77e7e', E: '#7ef7a0', W: '#f7d07e' };

export default function App() {
  const [filePath,       setFilePath]       = useState(null);
  const [isProcessing,   setIsProcessing]   = useState(null); // null | 'analysis' | 'roi'
  const [results,        setResults]        = useState(null);
  const [activePanel,    setActivePanel]    = useState('original');
  const [imgTs,          setImgTs]          = useState(0);   // cache-bust key

  // calibration
  const [calibMode,      setCalibMode]      = useState('idle'); // idle|pt1|pt2
  const [calibPt1,       setCalibPt1]       = useState(null);
  const [calibPt2,       setCalibPt2]       = useState(null);
  const [pxPerCm,        setPxPerCm]        = useState(null);
  const [calibDist,      setCalibDist]      = useState('10');
  const [showDistInput,  setShowDistInput]  = useState(false);

  // ROI
  const [roiMode,        setRoiMode]        = useState(false);
  const [roiRadius,      setRoiRadius]      = useState('5.0');
  const [roiCircle,      setRoiCircle]      = useState(null);
  const [roiResults,     setRoiResults]     = useState(null);

  const imgRef = useRef(null);

  // ── pixel coords from mouse event ─────────────────────────────────────────
  const getCoords = (e) => {
    const img  = imgRef.current;
    if (!img || !results) return null;
    const rect = img.getBoundingClientRect();
    const sx   = results.shape[1] / rect.width;
    const sy   = results.shape[0] / rect.height;
    return {
      px: Math.round((e.clientX - rect.left) * sx),
      py: Math.round((e.clientY - rect.top)  * sy),
      dx: ((e.clientX - rect.left) / rect.width)  * 100,
      dy: ((e.clientY - rect.top)  / rect.height) * 100,
    };
  };

  // ── file pick / drop ──────────────────────────────────────────────────────
  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFilePath(f.path);
  };

  const handleBrowse = async () => {
    const p = await ipcRenderer.invoke('open-file-dialog');
    if (p) setFilePath(p);
  };

  // ── run analysis ──────────────────────────────────────────────────────────
  const runAnalysis = async () => {
    if (!filePath) return;
    setIsProcessing('analysis');
    try {
      const outDir   = filePath + '_analysis';
      const response = await ipcRenderer.invoke('run-analysis', filePath, outDir);
      setResults(response);
      setImgTs(Date.now());
      setActivePanel('original');
      resetCalib();
      setRoiCircle(null);
      setRoiResults(null);
    } catch (err) {
      alert(`Analysis failed:\n${err}`);
    }
    setIsProcessing(null);
  };

  // ── calibration helpers ───────────────────────────────────────────────────
  const resetCalib = () => {
    setCalibMode('idle');
    setCalibPt1(null);
    setCalibPt2(null);
    setShowDistInput(false);
  };

  const startCalib = () => {
    resetCalib();
    setPxPerCm(null);
    setCalibMode('pt1');
  };

  const confirmCalib = () => {
    if (!calibPt1 || !calibPt2) return;
    const dist_px = Math.hypot(calibPt2.px - calibPt1.px, calibPt2.py - calibPt1.py);
    const real_cm = parseFloat(calibDist);
    if (!real_cm || real_cm <= 0) { alert('Enter a valid distance > 0'); return; }
    setPxPerCm(dist_px / real_cm);
    setShowDistInput(false);
    setCalibMode('idle');
  };

  // ── image click ───────────────────────────────────────────────────────────
  const handleImageClick = async (e) => {
    const c = getCoords(e);
    if (!c) return;

    if (calibMode === 'pt1') { setCalibPt1(c); setCalibMode('pt2'); return; }

    if (calibMode === 'pt2') {
      setCalibPt2(c);
      setCalibMode('idle');
      setShowDistInput(true);
      return;
    }

    if (roiMode) {
      if (!pxPerCm) { alert('Calibrate first!'); setRoiMode(false); return; }
      const r_cm = parseFloat(roiRadius) || 5.0;
      const r_px = r_cm * pxPerCm;
      setRoiCircle({
        dx: c.dx, dy: c.dy, px: c.px, py: c.py,
        rw: (r_px / results.shape[1]) * 100,
        rh: (r_px / results.shape[0]) * 100,
      });
      setRoiMode(false);
      setIsProcessing('roi');
      try {
        const roi = await ipcRenderer.invoke(
          'measure-roi', filePath,
          c.px, c.py, r_cm, pxPerCm, results.out_dir
        );
        setRoiResults(roi);
      } catch (err) {
        alert(`ROI failed:\n${err}`);
      }
      setIsProcessing(null);
    }
  };

  // ── open folder ───────────────────────────────────────────────────────────
  const openFolder = () => {
    const { shell } = window.require('electron');
    shell.openPath(results.out_dir);
  };

  const showCsv = (p) => {
    const { shell } = window.require('electron');
    shell.showItemInFolder(p);
  };

  // ── status bar text ───────────────────────────────────────────────────────
  const statusText = () => {
    if (calibMode === 'pt1')           return '📍 Click point 1 on the image';
    if (calibMode === 'pt2')           return '📍 Click point 2 on the image';
    if (roiMode)                       return `⊙ Click the centre of your region  (r = ${roiRadius} cm)`;
    if (isProcessing === 'roi')        return '⏳ Computing ROI…';
    if (pxPerCm)                       return `✓ Calibrated — ${pxPerCm.toFixed(2)} px/cm`;
    return 'Not calibrated — use the calibration tool on the right';
  };

  const cursor = (calibMode !== 'idle' || roiMode) ? 'crosshair' : 'default';

  const imgSrc = results?.images?.[activePanel]
    ? `${toFileUrl(results.images[activePanel])}?v=${imgTs}`
    : null;

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="app">

      {/* ══ HEADER ═══════════════════════════════════════════════════════ */}
      <header className="app-header">
        <div className="header-brand">
          <span className="brand-icon">🌡</span>
          <span className="brand-name">ThermalSight</span>
        </div>
        {results && (
          <button className="btn-ghost" onClick={() => { setResults(null); setFilePath(null); }}>
            ← New Image
          </button>
        )}
      </header>

      {/* ══ UPLOAD SCREEN ════════════════════════════════════════════════ */}
      {!results && (
        <main className="upload-screen">
          <div className="upload-hero">
            <h2>Thermal Gradient Analysis</h2>
            <p>Load a FLIR or thermal image to visualise temperature gradients, measure heat flow direction, and export data.</p>
          </div>

          <div className="drop-zone"
               onDragOver={(e) => e.preventDefault()}
               onDrop={handleDrop}
               onClick={handleBrowse}>
            <div className="drop-icon">📷</div>
            {filePath
              ? <p className="drop-selected">{filePath.split(/[\\/]/).pop()}<br /><span className="drop-path">{filePath}</span></p>
              : <>
                  <p className="drop-title">Drop a thermal image here</p>
                  <p className="drop-sub">or click to browse  ·  .jpg  .png  .tiff</p>
                </>
            }
          </div>

          <button className="btn-primary btn-xl"
                  disabled={!filePath || isProcessing === 'analysis'}
                  onClick={runAnalysis}>
            {isProcessing === 'analysis'
              ? <><span className="spinner" />  Analysing heat flow…</>
              : 'Analyse Image'}
          </button>
        </main>
      )}

      {/* ══ RESULTS SCREEN ═══════════════════════════════════════════════ */}
      {results && (
        <div className="results-layout">

          {/* ── LEFT: panel list ─────────────────────────────────────── */}
          <aside className="left-sidebar">
            <p className="sidebar-heading">PANELS</p>
            {PANELS.map(p => (
              <button key={p.key}
                      className={`panel-btn ${activePanel === p.key ? 'active' : ''}`}
                      title={p.desc}
                      onClick={() => setActivePanel(p.key)}>
                <span className="panel-icon">{p.icon}</span>
                <span className="panel-label">{p.label}</span>
              </button>
            ))}
            <div className="sidebar-sep" />
            <button className="panel-btn" onClick={openFolder}>
              <span className="panel-icon">📁</span>
              <span className="panel-label">Output folder</span>
            </button>
          </aside>

          {/* ── CENTRE: image + overlays ─────────────────────────────── */}
          <div className="image-area">
            <div className="image-wrapper">

              {imgSrc && (
                <img ref={imgRef}
                     src={imgSrc}
                     alt={activePanel}
                     className="thermal-img"
                     style={{ cursor }}
                     draggable={false}
                     onClick={handleImageClick} />
              )}

              {/* calib dots */}
              {calibPt1 && (
                <div className="ov-dot calib-dot" style={{ left:`${calibPt1.dx}%`, top:`${calibPt1.dy}%` }}>
                  1
                </div>
              )}
              {calibPt2 && (
                <div className="ov-dot calib-dot" style={{ left:`${calibPt2.dx}%`, top:`${calibPt2.dy}%` }}>
                  2
                </div>
              )}

              {/* calib line */}
              {calibPt1 && calibPt2 && (
                <svg className="ov-svg">
                  <line x1={`${calibPt1.dx}%`} y1={`${calibPt1.dy}%`}
                        x2={`${calibPt2.dx}%`} y2={`${calibPt2.dy}%`}
                        stroke="#00e5ff" strokeWidth="1.5" strokeDasharray="5 3" />
                </svg>
              )}

              {/* ROI circle */}
              {roiCircle && (
                <svg className="ov-svg">
                  <ellipse cx={`${roiCircle.dx}%`} cy={`${roiCircle.dy}%`}
                           rx={`${roiCircle.rw}%`} ry={`${roiCircle.rh}%`}
                           fill="none" stroke="#ffd54f" strokeWidth="2" />
                  <line x1={`${roiCircle.dx - roiCircle.rw}%`} y1={`${roiCircle.dy}%`}
                        x2={`${roiCircle.dx + roiCircle.rw}%`} y2={`${roiCircle.dy}%`}
                        stroke="#ffd54f" strokeWidth="0.8" strokeDasharray="4 3" />
                  <line x1={`${roiCircle.dx}%`} y1={`${roiCircle.dy - roiCircle.rh}%`}
                        x2={`${roiCircle.dx}%`} y2={`${roiCircle.dy + roiCircle.rh}%`}
                        stroke="#ffd54f" strokeWidth="0.8" strokeDasharray="4 3" />
                  {[['N',0,-1.4],['S',0,1.6],['E',1.4,0],['W',-1.4,0]].map(([l,ox,oy]) => (
                    <text key={l}
                          x={`${roiCircle.dx + ox*roiCircle.rw}%`}
                          y={`${roiCircle.dy + oy*roiCircle.rh}%`}
                          fill="#ffd54f" fontSize="11" fontWeight="bold"
                          textAnchor="middle" dominantBaseline="middle">{l}</text>
                  ))}
                </svg>
              )}

              {/* status bar */}
              <div className={`img-statusbar ${calibMode!=='idle'||roiMode ? 'active' : ''}`}>
                {statusText()}
              </div>

            </div>{/* image-wrapper */}
          </div>{/* image-area */}

          {/* ── RIGHT: tools ─────────────────────────────────────────── */}
          <aside className="right-sidebar">

            {/* image stats */}
            <div className="tool-card">
              <h4 className="card-title">Image Info</h4>
              <div className="kv"><span>File</span><span>{results.stem}</span></div>
              <div className="kv"><span>Size</span><span>{results.shape[1]} × {results.shape[0]} px</span></div>
              <div className="kv"><span>Temp min</span><span>{results.temp_min?.toFixed(1)} °C</span></div>
              <div className="kv"><span>Temp max</span><span>{results.temp_max?.toFixed(1)} °C</span></div>
              <div className="kv"><span>Temp mean</span><span>{results.temp_mean?.toFixed(1)} °C</span></div>
            </div>

            {/* calibration */}
            <div className="tool-card">
              <h4 className="card-title">📏 Calibration</h4>
              <p className={`calib-status ${pxPerCm ? 'ok' : 'none'}`}>
                {pxPerCm ? `✓ ${pxPerCm.toFixed(2)} px / cm` : 'Not calibrated'}
              </p>
              <button className="btn-secondary w-full" onClick={startCalib}>
                Click 2 points on image
              </button>

              {showDistInput && (
                <div className="inline-form">
                  <label>Real distance between points (cm):</label>
                  <input type="number" min="0.1" step="0.1"
                         value={calibDist}
                         autoFocus
                         onChange={e => setCalibDist(e.target.value)}
                         onKeyDown={e => { if (e.key==='Enter') confirmCalib(); if (e.key==='Escape') resetCalib(); }} />
                  <div className="inline-form-btns">
                    <button className="btn-primary" onClick={confirmCalib}>Confirm</button>
                    <button className="btn-ghost"   onClick={resetCalib}>Cancel</button>
                  </div>
                </div>
              )}
            </div>

            {/* ROI */}
            <div className="tool-card">
              <h4 className="card-title">⊙ Measure Region</h4>
              <label className="field-label">Radius (cm)</label>
              <input className="field-input" type="number" min="0.1" step="0.5"
                     value={roiRadius}
                     onChange={e => setRoiRadius(e.target.value)} />
              <button className={`btn-secondary w-full ${roiMode?'btn-active':''}`}
                      disabled={!pxPerCm || isProcessing === 'roi'}
                      onClick={() => setRoiMode(v => !v)}>
                {isProcessing==='roi' ? <><span className="spinner"/>Computing…</>
                  : roiMode ? '…click on image' : 'Place circle'}
              </button>
            </div>

            {/* ROI results */}
            {roiResults && (
              <div className="tool-card roi-card">
                <h4 className="card-title">ROI Results</h4>

                <div className="kv"><span>Centre</span>
                  <span>({roiResults.centre_px[0]}, {roiResults.centre_px[1]})</span></div>
                <div className="kv"><span>Radius</span>
                  <span>{roiResults.radius_cm} cm</span></div>
                <div className="kv highlight"><span>∇ / cm</span>
                  <span>{roiResults.grad_per_cm?.toFixed(4)}</span></div>
                <div className="kv"><span>Pixels</span>
                  <span>{roiResults.n_pixels?.toLocaleString()}</span></div>

                {/* directional bars */}
                <div className="dir-grid">
                  {['N','S','E','W'].map(d => {
                    const val = roiResults.directional[d] ?? 0;
                    const max = Math.max(...['N','S','E','W'].map(x => roiResults.directional[x]??0));
                    const pct = max > 0 ? (val/max)*100 : 0;
                    return (
                      <div key={d} className={`dir-row ${roiResults.dominant===d?'dominant':''}`}>
                        <span className="dir-lbl" style={{color:DIR_COLORS[d]}}>{d}</span>
                        <div className="dir-bar-wrap">
                          <div className="dir-bar" style={{width:`${pct}%`, background:DIR_COLORS[d]}} />
                        </div>
                        <span className="dir-num">{val.toFixed(3)}</span>
                      </div>
                    );
                  })}
                </div>

                <div className="kv muted"><span>Temp mean</span>
                  <span>{roiResults.stats?.temp?.mean?.toFixed(2)} °C</span></div>
                <div className="kv muted"><span>Temp std</span>
                  <span>± {roiResults.stats?.temp?.std?.toFixed(2)} °C</span></div>
                <div className="kv muted"><span>Net E→W</span>
                  <span>{roiResults.directional?.net_x?.toFixed(3)}</span></div>
                <div className="kv muted"><span>Net N→S</span>
                  <span>{roiResults.directional?.net_y?.toFixed(3)}</span></div>

                <button className="btn-secondary w-full" style={{marginTop:'8px'}}
                        onClick={() => showCsv(roiResults.csv_path)}>
                  📄 Show CSV in folder
                </button>
              </div>
            )}

          </aside>
        </div>
      )}
    </div>
  );
}
