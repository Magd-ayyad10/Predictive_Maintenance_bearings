import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine } from 'recharts';

function App() {
  // Navigation State - 6 tabs like app.py
  const [activeTab, setActiveTab] = useState("overview");

  // --- LIVE MONITOR STATE ---
  const [status, setStatus] = useState("WAITING");
  const [score, setScore] = useState(0);
  const [threshold, setThreshold] = useState(0);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [simulationMessage, setSimulationMessage] = useState(null); // For completion message

  // --- ENHANCED SIMULATION STATE ---
  const [simulationSpeed, setSimulationSpeed] = useState(100); // ms between updates
  const [totalReadings, setTotalReadings] = useState(0);
  const [anomalyCount, setAnomalyCount] = useState(0);
  const [alertLog, setAlertLog] = useState([]); // Recent alerts
  const [maxScore, setMaxScore] = useState(0);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [totalDataPoints, setTotalDataPoints] = useState(100); // Will be updated from API
  const [healthCheckResult, setHealthCheckResult] = useState(null);
  const [healthCheckLoading, setHealthCheckLoading] = useState(false);

  // --- BATCH ANALYSIS STATE ---
  const [batchFile, setBatchFile] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // --- CUSTOM SIMULATION STATE (for uploaded data) ---
  const [customDataLoaded, setCustomDataLoaded] = useState(false);
  const [customTotalRecords, setCustomTotalRecords] = useState(0);
  const [customTransformInfo, setCustomTransformInfo] = useState(null);
  const [isCustomSimulating, setIsCustomSimulating] = useState(false);
  const [customSimulationHistory, setCustomSimulationHistory] = useState([]);
  const [customCurrentIndex, setCustomCurrentIndex] = useState(0);
  const [customScore, setCustomScore] = useState(0);
  const [customStatus, setCustomStatus] = useState("WAITING");
  const [customAnomalyCount, setCustomAnomalyCount] = useState(0);
  const [customMaxScore, setCustomMaxScore] = useState(0);
  const [customSimSpeed, setCustomSimSpeed] = useState(100);
  const customSpeedRef = useRef(100);
  const isCustomRunningRef = useRef(false);

  // --- MODEL INFO STATE ---
  const [modelInfo, setModelInfo] = useState({ threshold: 0.0005 });

  // Feature importance data for Diagnostics
  const featureImportance = [
    { name: 'B1_rms', importance: 0.15 },
    { name: 'B1_kurt', importance: 0.12 },
    { name: 'B1_skew', importance: 0.08 },
    { name: 'B1_peak', importance: 0.14 },
    { name: 'B1_crest', importance: 0.09 },
    { name: 'B2_rms', importance: 0.13 },
    { name: 'B2_kurt', importance: 0.11 },
    { name: 'B2_skew', importance: 0.07 },
    { name: 'B2_peak', importance: 0.16 },
    { name: 'B2_crest', importance: 0.10 },
    { name: 'B3_rms', importance: 0.12 },
    { name: 'B3_kurt', importance: 0.09 },
    { name: 'B3_skew', importance: 0.06 },
    { name: 'B3_peak', importance: 0.11 },
    { name: 'B3_crest', importance: 0.08 },
    { name: 'B4_rms', importance: 0.14 },
    { name: 'B4_kurt', importance: 0.10 },
    { name: 'B4_skew', importance: 0.05 },
    { name: 'B4_peak', importance: 0.13 },
    { name: 'B4_crest', importance: 0.07 },
  ].sort((a, b) => a.importance - b.importance);

  // Tab configuration
  const tabs = [
    { id: "overview", label: "🏠 Overview" },
    { id: "guide", label: "📚 Project Guide" },
    { id: "live", label: "📡 Live Simulation" },
    { id: "batch", label: "📂 Batch Analysis" },
    { id: "diagnostics", label: "🔍 Diagnostics" },
    { id: "modelinfo", label: "📊 Model Info" },
  ];

  // ==========================
  // 1. LIVE SIMULATION LOGIC
  // ==========================
  // --- NEW: REPLAY MODE ---
  const [isReplaying, setIsReplaying] = useState(false);
  const intervalRef = useRef(null); // Store interval reference to prevent duplicates
  const isRunningRef = useRef(false); // Track running state for stop functionality

  // Fetch model info (threshold) on mount
  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/model-info');
        setModelInfo(response.data);
        setThreshold(response.data.threshold);
      } catch (error) {
        console.error("Could not fetch model info:", error);
      }
    };
    fetchModelInfo();
  }, []);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Stop simulation function
  const stopSimulation = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    isRunningRef.current = false;
    setIsReplaying(false);
    setLoading(false);
  };

  // Reset all simulation data
  const resetSimulation = () => {
    setHistory([]);
    setTotalReadings(0);
    setAnomalyCount(0);
    setAlertLog([]);
    setMaxScore(0);
    setCurrentIndex(0);
    setScore(0);
    setStatus("WAITING");
    setSimulationMessage(null);
    setHealthCheckResult(null);
  };

  // Health Check function
  const runHealthCheck = async () => {
    setHealthCheckLoading(true);
    try {
      const response = await axios.get('http://127.0.0.1:8000/health-check');
      setHealthCheckResult(response.data);
    } catch (error) {
      setHealthCheckResult({
        overall: "ERROR",
        database: "CONNECTION_FAILED",
        model: "UNKNOWN",
        predictions_table: "UNKNOWN",
        total_predictions: 0,
        error: error.message
      });
    }
    setHealthCheckLoading(false);
  };

  // Speed reference for dynamic speed changes
  const speedRef = useRef(simulationSpeed);
  useEffect(() => {
    speedRef.current = simulationSpeed;
  }, [simulationSpeed]);

  // This function loops automatically!
  const startReplay = async () => {
    // If already running, stop it
    if (isRunningRef.current) {
      stopSimulation();
      return;
    }

    // Start the simulation
    isRunningRef.current = true;
    setIsReplaying(true);
    setLoading(true);
    setSimulationMessage(null); // Clear any previous message

    // Reset stats for new run
    setTotalReadings(0);
    setAnomalyCount(0);
    setAlertLog([]);
    setMaxScore(0);
    setCurrentIndex(0);

    // 1. Reset the backend index to 0 and get total records
    const resetResponse = await axios.post('http://127.0.0.1:8000/simulate/reset');
    if (resetResponse.data.total_records) {
      setTotalDataPoints(resetResponse.data.total_records);
    }

    // 2. Start the Loop with dynamic speed
    const runNext = async () => {
      // Check if we should stop
      if (!isRunningRef.current) {
        return;
      }

      try {
        const response = await axios.get('http://127.0.0.1:8000/simulate/next');
        const data = response.data;

        if (data.finished) {
          isRunningRef.current = false;
          setIsReplaying(false);
          setLoading(false);
          setStatus("SIMULATION_COMPLETE");
          setSimulationMessage("🔔 Simulation Complete! The bearing has failed.");
          return;
        }

        // Update UI
        setScore(data.anomaly_score);
        setThreshold(data.threshold);
        setStatus(data.status);
        setCurrentIndex(data.index);
        setTotalReadings(prev => prev + 1);

        // Track max score
        setMaxScore(prev => Math.max(prev, data.anomaly_score));

        // Track anomalies and log alerts
        if (data.status === "CRITICAL_FAILURE") {
          setAnomalyCount(prev => prev + 1);
          setAlertLog(prev => [{
            time: new Date().toLocaleTimeString(),
            index: data.index,
            score: data.anomaly_score.toFixed(4),
            type: "CRITICAL"
          }, ...prev].slice(0, 10)); // Keep last 10 alerts
        }

        // Add to Graph
        setHistory(prev => {
          const newPoint = {
            // Use the REAL timestamp from NASA, simplified
            time: data.index,
            score: data.anomaly_score,
            limit: data.threshold
          };
          return [...prev.slice(-49), newPoint]; // Keep last 50 points
        });

        // Schedule next with current speed
        if (isRunningRef.current) {
          setTimeout(runNext, speedRef.current);
        }

      } catch (err) {
        console.error("Simulation Error", err);
        isRunningRef.current = false;
        setIsReplaying(false);
        setLoading(false);
      }
    };

    // Start the recursive loop
    runNext();
  };

  // ==========================
  // 2. BATCH UPLOAD LOGIC
  // ==========================
  const handleFileChange = (e) => {
    setBatchFile(e.target.files[0]);
  };

  const uploadBatch = async () => {
    if (!batchFile) return alert("Please select a CSV file first!");

    setBatchLoading(true);
    const formData = new FormData();
    formData.append("file", batchFile);

    try {
      const response = await axios.post('http://127.0.0.1:8000/analyze-batch', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setBatchResults(response.data);
    } catch (error) {
      console.error("Upload Error:", error);
      alert("Failed to analyze file. Check Python console.");
    }
    setBatchLoading(false);
  };

  // Styles
  const styles = {
    app: {
      backgroundColor: "#0e1117",
      color: "#fafafa",
      minHeight: "100vh",
      padding: "20px 40px",
      fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    },
    header: {
      background: "linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%)",
      padding: "25px 30px",
      borderRadius: "12px",
      marginBottom: "25px"
    },
    headerTitle: {
      margin: 0,
      color: "#000",
      fontSize: "28px",
      fontWeight: "700"
    },
    headerSubtitle: {
      margin: "8px 0 0 0",
      color: "#333",
      fontSize: "14px"
    },
    tabContainer: {
      display: "flex",
      gap: "8px",
      marginBottom: "25px",
      flexWrap: "wrap"
    },
    tab: (isActive) => ({
      padding: "12px 20px",
      backgroundColor: isActive ? "#262730" : "transparent",
      color: isActive ? "#00C9FF" : "#888",
      border: isActive ? "1px solid #00C9FF" : "1px solid #464b5c",
      borderRadius: "8px",
      cursor: "pointer",
      fontWeight: isActive ? "600" : "400",
      transition: "all 0.2s ease"
    }),
    card: {
      backgroundColor: "#262730",
      border: "1px solid #464b5c",
      padding: "20px",
      borderRadius: "12px",
      marginBottom: "20px"
    },
    metricCard: {
      backgroundColor: "#262730",
      border: "1px solid #464b5c",
      padding: "20px",
      borderRadius: "12px",
      textAlign: "center"
    },
    metricValue: {
      fontSize: "28px",
      fontWeight: "700",
      color: "#00C9FF",
      margin: "10px 0"
    },
    metricLabel: {
      fontSize: "14px",
      color: "#888"
    },
    metricDelta: {
      fontSize: "12px",
      color: "#92FE9D"
    },
    grid3: {
      display: "grid",
      gridTemplateColumns: "repeat(3, 1fr)",
      gap: "20px",
      marginBottom: "25px"
    },
    grid2: {
      display: "grid",
      gridTemplateColumns: "repeat(2, 1fr)",
      gap: "20px",
      marginBottom: "25px"
    },
    button: {
      padding: "12px 24px",
      backgroundColor: "#00C9FF",
      color: "#000",
      border: "none",
      borderRadius: "8px",
      cursor: "pointer",
      fontWeight: "600",
      fontSize: "14px"
    },
    buttonSecondary: {
      padding: "12px 24px",
      backgroundColor: "#ff4b4b",
      color: "#fff",
      border: "none",
      borderRadius: "8px",
      cursor: "pointer",
      fontWeight: "600",
      fontSize: "14px"
    },
    select: {
      padding: "10px 15px",
      backgroundColor: "#1e1e2e",
      color: "#fff",
      border: "1px solid #464b5c",
      borderRadius: "8px",
      fontSize: "14px",
      marginRight: "10px"
    },
    table: {
      width: "100%",
      borderCollapse: "collapse",
      marginTop: "15px"
    },
    th: {
      textAlign: "left",
      padding: "12px",
      borderBottom: "1px solid #464b5c",
      color: "#888"
    },
    td: {
      padding: "12px",
      borderBottom: "1px solid #333"
    }
  };

  return (
    <div style={styles.app}>

      {/* HEADER */}
      <div style={styles.header}>
        <h1 style={styles.headerTitle}>⚙️ Universal Bearing Guard AI</h1>
        <p style={styles.headerSubtitle}>Predictive Maintenance System | Trained on NASA IMS Sets 1, 2, & 3</p>
      </div>

      {/* TABS NAVIGATION */}
      <div style={styles.tabContainer}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={styles.tab(activeTab === tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ================= TAB 1: OVERVIEW ================= */}
      {activeTab === "overview" && (
        <div>
          <div style={styles.grid3}>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>System Status</div>
              <div style={styles.metricValue}>Online</div>
              <div style={styles.metricDelta}>Universal Model Active</div>
            </div>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Anomaly Threshold</div>
              <div style={styles.metricValue}>{modelInfo.threshold.toFixed(4)}</div>
              <div style={styles.metricDelta}>Based on Universal Healthy Baseline</div>
            </div>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Training Data</div>
              <div style={styles.metricValue}>NASA Sets 1-3</div>
              <div style={styles.metricDelta}>Multi-Condition</div>
            </div>
          </div>

          <div style={styles.card}>
            <h2>How it Works</h2>
            <p>This system uses <strong>Unsupervised Learning (PCA)</strong> to detect bearing failures before they happen.</p>
            <ol>
              <li><strong>Universal Baseline:</strong> The model learned "Normal" vibration patterns from three different experiments.</li>
              <li><strong>Detection Logic:</strong> If the Reconstruction Error exceeds <strong>{modelInfo.threshold.toFixed(4)}</strong>, an alarm is triggered.</li>
              <li><strong>Root Cause:</strong> The system automatically identifies which sensor (Bearing 1, 2, 3, or 4) is causing the anomaly.</li>
            </ol>
          </div>

          <h3>Quick Start</h3>
          <div style={styles.grid3}>
            <div style={styles.card}>
              <h4>1️⃣ Real-Time Demo</h4>
              <p style={{ color: "#888" }}>Test live sensor streams with different failure conditions.</p>
            </div>
            <div style={styles.card}>
              <h4>2️⃣ Upload Data</h4>
              <p style={{ color: "#888" }}>Analyze your own bearing vibration datasets.</p>
            </div>
            <div style={styles.card}>
              <h4>3️⃣ Diagnose Issues</h4>
              <p style={{ color: "#888" }}>Identify which bearing is failing with root cause analysis.</p>
            </div>
          </div>
        </div>
      )}

      {/* ================= TAB 2: PROJECT GUIDE ================= */}
      {activeTab === "guide" && (
        <div>
          <div style={styles.grid2}>
            <div style={styles.card}>
              <h2>What is Predictive Maintenance?</h2>
              <p>Instead of replacing bearings when they break (<strong>reactive</strong>) or on a schedule (<strong>preventive</strong>), this system <strong>predicts failures before they happen</strong> using machine learning on vibration data.</p>
              <h4>Key Benefits:</h4>
              <ul>
                <li>🛡️ Prevent unexpected downtime</li>
                <li>💰 Reduce maintenance costs</li>
                <li>⚡ Optimize equipment lifespan</li>
                <li>📊 Data-driven decision making</li>
              </ul>
            </div>
            <div style={{ ...styles.card, backgroundColor: "#1a3a4a" }}>
              <h3>Why Bearings?</h3>
              <p>Bearings are critical rotating components that fail predictably. Vibration patterns reveal degradation weeks before failure.</p>
            </div>
          </div>

          <div style={styles.grid2}>
            <div style={styles.card}>
              <h3>Features Used (20 Total)</h3>
              <p>For each of 4 bearings, we extract:</p>
              <ul>
                <li><strong>RMS:</strong> Overall vibration energy</li>
                <li><strong>Kurtosis:</strong> Spikiness (impulsiveness)</li>
                <li><strong>Skewness:</strong> Asymmetry of distribution</li>
                <li><strong>Peak:</strong> Maximum amplitude</li>
                <li><strong>Crest Factor:</strong> Peak-to-RMS ratio</li>
              </ul>
            </div>
            <div style={styles.card}>
              <h3>Model Architecture</h3>
              <p><strong>Algorithm:</strong> Principal Component Analysis (PCA)</p>
              <p><strong>Approach:</strong> Unsupervised anomaly detection</p>
              <ul>
                <li>Learns "normal" from healthy operation</li>
                <li>Detects deviations via reconstruction error</li>
                <li>No labeled failure data needed</li>
              </ul>
            </div>
          </div>

          <div style={styles.card}>
            <h3>🎯 How the System Works</h3>
            <div style={styles.grid2}>
              <div>
                <p><strong>1️⃣ Collect</strong> → Raw vibration signals from sensors at ~20kHz</p>
                <p><strong>2️⃣ Extract</strong> → Compute 5 statistical features per bearing (20 total)</p>
                <p><strong>3️⃣ Normalize</strong> → Scale features using StandardScaler for model</p>
                <p><strong>4️⃣ Transform</strong> → PCA projects data to lower dimension space</p>
              </div>
              <div>
                <p><strong>5️⃣ Reconstruct</strong> → Model regenerates features from PCA components</p>
                <p><strong>6️⃣ Detect</strong> → Compare original vs reconstructed → Anomaly Score</p>
                <p><strong>7️⃣ Alert</strong> → If score &gt; threshold, trigger alarm</p>
              </div>
            </div>
          </div>

          <div style={styles.card}>
            <h3>🔗 Typical Bearing Failure Progression</h3>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Phase</th>
                  <th style={styles.th}>Timeline</th>
                  <th style={styles.th}>Symptoms</th>
                  <th style={styles.th}>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr><td style={styles.td}><strong>Healthy</strong></td><td style={styles.td}>Months</td><td style={styles.td}>Normal vibration, low variance</td><td style={styles.td}>✅ Green</td></tr>
                <tr><td style={styles.td}><strong>Early Degradation</strong></td><td style={styles.td}>Weeks</td><td style={styles.td}>Slight energy increase, kurtosis rises</td><td style={styles.td}>🟡 Yellow</td></tr>
                <tr><td style={styles.td}><strong>Advanced Decay</strong></td><td style={styles.td}>Days</td><td style={styles.td}>High spikes, peak values spike</td><td style={styles.td}>🟠 Orange</td></tr>
                <tr><td style={styles.td}><strong>Critical Failure</strong></td><td style={styles.td}>Hours</td><td style={styles.td}>Massive reconstruction error</td><td style={styles.td}>🚨 Red</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ================= TAB 3: LIVE SIMULATION ================= */}
      {activeTab === "live" && (
        <div>
          {/* Statistics Row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "15px", marginBottom: "20px" }}>
            <div style={{ ...styles.metricCard, padding: "15px" }}>
              <div style={styles.metricLabel}>Total Readings</div>
              <div style={{ ...styles.metricValue, fontSize: "24px" }}>{totalReadings}</div>
              <div style={styles.metricDelta}>of {totalDataPoints}</div>
            </div>
            <div style={{ ...styles.metricCard, padding: "15px" }}>
              <div style={styles.metricLabel}>Anomalies Detected</div>
              <div style={{ ...styles.metricValue, fontSize: "24px", color: anomalyCount > 0 ? "#dc3545" : "#28a745" }}>{anomalyCount}</div>
              <div style={styles.metricDelta}>{totalReadings > 0 ? ((anomalyCount / totalReadings) * 100).toFixed(1) : 0}% failure rate</div>
            </div>
            <div style={{ ...styles.metricCard, padding: "15px" }}>
              <div style={styles.metricLabel}>Max Score</div>
              <div style={{ ...styles.metricValue, fontSize: "24px", color: maxScore > threshold ? "#dc3545" : "#28a745" }}>{maxScore.toFixed(4)}</div>
              <div style={styles.metricDelta}>Peak anomaly detected</div>
            </div>
            <div style={{ ...styles.metricCard, padding: "15px" }}>
              <div style={styles.metricLabel}>Health Status</div>
              <div style={{ ...styles.metricValue, fontSize: "24px", color: anomalyCount === 0 ? "#28a745" : anomalyCount < 5 ? "#ffc107" : "#dc3545" }}>
                {anomalyCount === 0 ? "100%" : Math.max(0, (100 - (anomalyCount / totalReadings) * 100)).toFixed(0) + "%"}
              </div>
              <div style={styles.metricDelta}>System health</div>
            </div>
          </div>

          {/* Progress Bar */}
          <div style={{ ...styles.card, padding: "15px", marginBottom: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ color: "#888", fontSize: "14px" }}>Simulation Progress</span>
              <span style={{ color: "#00C9FF", fontWeight: "600" }}>{currentIndex} / {totalDataPoints}</span>
            </div>
            <div style={{
              height: "8px",
              backgroundColor: "#333",
              borderRadius: "4px",
              overflow: "hidden"
            }}>
              <div style={{
                height: "100%",
                width: `${(currentIndex / totalDataPoints) * 100}%`,
                background: "linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%)",
                borderRadius: "4px",
                transition: "width 0.1s ease"
              }} />
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "30px" }}>

            {/* Control Panel */}
            <div style={styles.card}>
              <h3 style={{ borderBottom: "1px solid #464b5c", paddingBottom: "10px" }}>Sensor Status</h3>

              <div style={{
                padding: "20px",
                borderRadius: "8px",
                textAlign: "center",
                marginBottom: "20px",
                backgroundColor: status === "HEALTHY" ? "rgba(40, 167, 69, 0.2)" : status === "WAITING" ? "#333" : "rgba(220, 53, 69, 0.2)",
                border: status === "HEALTHY" ? "1px solid #28a745" : status === "WAITING" ? "1px solid #555" : "1px solid #dc3545",
                color: status === "HEALTHY" ? "#28a745" : status === "WAITING" ? "#888" : "#dc3545"
              }}>
                <h2 style={{ margin: 0 }}>{status}</h2>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "20px" }}>
                <div>
                  <small style={{ color: "#888" }}>Current Anomaly Score</small>
                  <div style={{ fontSize: "32px", fontWeight: "bold" }}>{score.toFixed(4)}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <small style={{ color: "#888" }}>Safety Threshold</small>
                  <div style={{ fontSize: "32px", color: "#ffc107" }}>{threshold.toFixed(4)}</div>
                </div>
              </div>

              {/* Speed Control */}
              <div style={{ marginBottom: "20px", padding: "15px", backgroundColor: "#1e1e2e", borderRadius: "8px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px" }}>
                  <span style={{ color: "#888", fontSize: "14px" }}>⚡ Simulation Speed</span>
                  <span style={{ color: "#00C9FF", fontWeight: "600" }}>
                    {simulationSpeed === 25 ? "Very Fast" : simulationSpeed === 50 ? "Fast" : simulationSpeed === 100 ? "Normal" : simulationSpeed === 200 ? "Slow" : `${simulationSpeed}ms`}
                  </span>
                </div>
                <input
                  type="range"
                  min="25"
                  max="500"
                  step="50"
                  value={simulationSpeed}
                  onChange={(e) => setSimulationSpeed(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "#00C9FF" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#666", marginTop: "5px" }}>
                  <span>Fast</span>
                  <span>Slow</span>
                </div>
              </div>

              <div style={{ display: "flex", gap: "10px", marginBottom: "10px" }}>

                {!isReplaying ? (
                  <button
                    onClick={startReplay}
                    disabled={loading}
                    style={{
                      padding: "16px 32px",
                      fontSize: "16px",
                      fontWeight: "600",
                      border: "none",
                      borderRadius: "12px",
                      cursor: loading ? "not-allowed" : "pointer",
                      background: "linear-gradient(135deg, #007bff 0%, #0056b3 100%)",
                      color: "#fff",
                      boxShadow: "0 4px 15px rgba(0, 123, 255, 0.4)",
                      transition: "all 0.3s ease",
                      width: "100%",
                      letterSpacing: "0.5px",
                      opacity: loading ? 0.7 : 1
                    }}
                  >
                    ▶ START REAL DATA REPLAY
                  </button>
                ) : (
                  <button
                    onClick={stopSimulation}
                    style={{
                      padding: "16px 32px",
                      fontSize: "16px",
                      fontWeight: "600",
                      border: "none",
                      borderRadius: "12px",
                      cursor: "pointer",
                      background: "linear-gradient(135deg, #dc3545 0%, #c82333 100%)",
                      color: "#fff",
                      boxShadow: "0 4px 15px rgba(220, 53, 69, 0.4)",
                      transition: "all 0.3s ease",
                      width: "100%",
                      letterSpacing: "0.5px"
                    }}
                  >
                    ⏹ STOP SIMULATION
                  </button>
                )}
              </div>

              {/* Reset Button */}
              <button
                onClick={resetSimulation}
                disabled={isReplaying}
                style={{
                  padding: "10px",
                  fontSize: "14px",
                  fontWeight: "500",
                  border: "1px solid #464b5c",
                  borderRadius: "8px",
                  cursor: isReplaying ? "not-allowed" : "pointer",
                  background: "transparent",
                  color: "#888",
                  width: "100%",
                  opacity: isReplaying ? 0.5 : 1,
                  transition: "all 0.2s ease"
                }}
              >
                🔄 Reset All Data
              </button>

              {/* Health Check Button */}
              <button
                onClick={runHealthCheck}
                disabled={healthCheckLoading}
                style={{
                  padding: "10px",
                  fontSize: "14px",
                  fontWeight: "500",
                  border: "1px solid #17a2b8",
                  borderRadius: "8px",
                  cursor: healthCheckLoading ? "not-allowed" : "pointer",
                  background: "rgba(23, 162, 184, 0.1)",
                  color: "#17a2b8",
                  width: "100%",
                  marginTop: "10px",
                  transition: "all 0.2s ease"
                }}
              >
                {healthCheckLoading ? "🔍 Checking..." : "🩺 Diagnose System Health"}
              </button>

              {/* Health Check Results */}
              {healthCheckResult && (
                <div style={{
                  marginTop: "15px",
                  padding: "15px",
                  backgroundColor: healthCheckResult.overall === "HEALTHY" ? "rgba(40, 167, 69, 0.1)" : "rgba(220, 53, 69, 0.1)",
                  border: `1px solid ${healthCheckResult.overall === "HEALTHY" ? "#28a745" : "#dc3545"}`,
                  borderRadius: "8px",
                  fontSize: "13px"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px" }}>
                    <strong style={{ color: healthCheckResult.overall === "HEALTHY" ? "#28a745" : "#dc3545" }}>
                      {healthCheckResult.overall === "HEALTHY" ? "✅ System Healthy" : "⚠️ Issues Detected"}
                    </strong>
                    <button onClick={() => setHealthCheckResult(null)} style={{ background: "none", border: "none", color: "#888", cursor: "pointer" }}>✕</button>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                    <div>📦 Model: <span style={{ color: healthCheckResult.model === "OK" ? "#28a745" : "#dc3545" }}>{healthCheckResult.model}</span></div>
                    <div>🗄️ Database: <span style={{ color: healthCheckResult.database === "OK" ? "#28a745" : "#dc3545" }}>{healthCheckResult.database}</span></div>
                    <div>📊 Predictions Table: <span style={{ color: healthCheckResult.predictions_table === "OK" ? "#28a745" : "#dc3545" }}>{healthCheckResult.predictions_table}</span></div>
                    <div>📈 Total Saved: <span style={{ color: "#00C9FF" }}>{healthCheckResult.total_predictions}</span></div>
                    <div>🎬 Simulation Data: <span style={{ color: healthCheckResult.simulation_data === "OK" ? "#28a745" : "#dc3545" }}>{healthCheckResult.simulation_data}</span></div>
                    <div>📂 Records Loaded: <span style={{ color: "#00C9FF" }}>{healthCheckResult.simulation_records}</span></div>
                  </div>

                  {healthCheckResult.last_predictions && healthCheckResult.last_predictions.length > 0 && (
                    <div style={{ marginTop: "12px", borderTop: "1px solid #464b5c", paddingTop: "10px" }}>
                      <div style={{ color: "#888", marginBottom: "6px" }}>Last {healthCheckResult.last_predictions.length} Predictions:</div>
                      {healthCheckResult.last_predictions.map((p, i) => (
                        <div key={i} style={{
                          padding: "4px 8px",
                          fontSize: "11px",
                          background: "#1e1e2e",
                          borderRadius: "4px",
                          marginBottom: "4px",
                          display: "flex",
                          justifyContent: "space-between"
                        }}>
                          <span>ID: {p.id}</span>
                          <span style={{ color: p.status === "HEALTHY" ? "#28a745" : "#dc3545" }}>{p.status}</span>
                          <span>Score: {p.score?.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {/* Simulation Complete Notification */}
              {simulationMessage && (
                <div style={{
                  marginTop: "15px",
                  padding: "15px",
                  backgroundColor: "rgba(255, 193, 7, 0.2)",
                  border: "1px solid #ffc107",
                  borderRadius: "8px",
                  color: "#ffc107",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center"
                }}>
                  <span>{simulationMessage}</span>
                  <button
                    onClick={() => setSimulationMessage(null)}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "#ffc107",
                      fontSize: "18px",
                      cursor: "pointer"
                    }}
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>

            {/* Right Column: Graph + Alert Log */}
            <div>
              {/* Live Graph */}
              <div style={{ ...styles.card, height: "300px", marginBottom: "20px" }}>
                <h3 style={{ borderBottom: "1px solid #464b5c", paddingBottom: "10px", marginBottom: "15px" }}>Vibration Anomaly Trend</h3>
                <ResponsiveContainer width="100%" height="80%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="time" stroke="#666" />
                    <YAxis stroke="#666" />
                    <Tooltip contentStyle={{ backgroundColor: "#333", border: "1px solid #555" }} />
                    <ReferenceLine y={threshold} label="Limit" stroke="red" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="score" stroke="#00C9FF" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Alert Log */}
              <div style={{ ...styles.card, maxHeight: "200px", overflow: "hidden" }}>
                <h3 style={{ borderBottom: "1px solid #464b5c", paddingBottom: "10px", marginBottom: "10px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span>🚨 Alert Log</span>
                  <span style={{ fontSize: "12px", color: "#888", fontWeight: "normal" }}>{alertLog.length} alerts</span>
                </h3>
                <div style={{ maxHeight: "130px", overflowY: "auto" }}>
                  {alertLog.length === 0 ? (
                    <div style={{ color: "#666", textAlign: "center", padding: "20px" }}>
                      No alerts yet. Start the simulation to detect anomalies.
                    </div>
                  ) : (
                    alertLog.map((alert, idx) => (
                      <div key={idx} style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "8px 10px",
                        marginBottom: "5px",
                        backgroundColor: "rgba(220, 53, 69, 0.1)",
                        borderRadius: "6px",
                        borderLeft: "3px solid #dc3545"
                      }}>
                        <span style={{ color: "#dc3545", fontWeight: "600" }}>⚠️ {alert.type}</span>
                        <span style={{ color: "#888" }}>Index: {alert.index}</span>
                        <span style={{ color: "#ffc107" }}>Score: {alert.score}</span>
                        <span style={{ color: "#666", fontSize: "12px" }}>{alert.time}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ================= TAB 4: BATCH ANALYSIS (ENHANCED) ================= */}
      {activeTab === "batch" && (
        <div>
          {/* Upload Section */}
          <div style={styles.card}>
            <h2>📂 Upload Any Sensor Data</h2>
            <p style={{ color: "#888", marginBottom: "20px" }}>
              Upload any CSV file - the AI will automatically transform it to the required format and run a live simulation.
              <br /><span style={{ color: "#00C9FF" }}>Supports: Raw signals, partial features, or any column configuration!</span>
            </p>

            <div style={{ border: "2px dashed #00C9FF", padding: "30px", borderRadius: "12px", backgroundColor: "#1a2a3a", textAlign: "center" }}>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                style={{ color: "#fff", marginBottom: "15px" }}
              />
              <br />
              <button
                onClick={async () => {
                  if (!batchFile) return alert("Please select a CSV file first!");
                  setBatchLoading(true);
                  const formData = new FormData();
                  formData.append("file", batchFile);
                  try {
                    const response = await axios.post('http://127.0.0.1:8000/preprocess-upload', formData, {
                      headers: { 'Content-Type': 'multipart/form-data' }
                    });
                    setCustomTransformInfo(response.data);
                    setCustomTotalRecords(response.data.transformed_rows);
                    setCustomDataLoaded(true);
                    setCustomSimulationHistory([]);
                    setCustomCurrentIndex(0);
                    setCustomAnomalyCount(0);
                    setCustomMaxScore(0);
                    setCustomScore(0);
                    setCustomStatus("READY");
                  } catch (error) {
                    console.error("Upload Error:", error);
                    alert("Failed to process file: " + (error.response?.data?.detail || error.message));
                  }
                  setBatchLoading(false);
                }}
                disabled={batchLoading || !batchFile}
                style={{
                  padding: "14px 28px",
                  fontSize: "16px",
                  fontWeight: "600",
                  border: "none",
                  borderRadius: "10px",
                  cursor: batchLoading ? "not-allowed" : "pointer",
                  background: "linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%)",
                  color: "#000",
                  boxShadow: "0 4px 15px rgba(0, 201, 255, 0.3)"
                }}
              >
                {batchLoading ? "🔄 Processing..." : "⬆️ Upload & Transform"}
              </button>
            </div>
          </div>

          {/* Transformation Info */}
          {customTransformInfo && (
            <div style={{ ...styles.card, marginTop: "20px", background: "linear-gradient(135deg, #1a3a4a 0%, #262730 100%)" }}>
              <h3 style={{ color: "#92FE9D", marginBottom: "15px" }}>✅ Data Transformed Successfully!</h3>
              <div style={styles.grid3}>
                <div style={styles.metricCard}>
                  <div style={styles.metricLabel}>Original Shape</div>
                  <div style={{ ...styles.metricValue, fontSize: "20px" }}>
                    {customTransformInfo.original_rows} × {customTransformInfo.original_columns}
                  </div>
                </div>
                <div style={styles.metricCard}>
                  <div style={styles.metricLabel}>Transformation</div>
                  <div style={{ ...styles.metricValue, fontSize: "18px", color: "#ffc107" }}>
                    {customTransformInfo.transformation_type?.replace('_', ' ').toUpperCase()}
                  </div>
                </div>
                <div style={styles.metricCard}>
                  <div style={styles.metricLabel}>Ready for Simulation</div>
                  <div style={{ ...styles.metricValue, fontSize: "20px" }}>
                    {customTransformInfo.transformed_rows} × 20
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Live Simulation for Custom Data */}
          {customDataLoaded && (
            <div style={{ marginTop: "20px" }}>
              {/* Stats Row */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "15px", marginBottom: "20px" }}>
                <div style={{ ...styles.metricCard, padding: "15px" }}>
                  <div style={styles.metricLabel}>Progress</div>
                  <div style={{ ...styles.metricValue, fontSize: "22px" }}>{customCurrentIndex}/{customTotalRecords}</div>
                </div>
                <div style={{ ...styles.metricCard, padding: "15px" }}>
                  <div style={styles.metricLabel}>Anomalies Found</div>
                  <div style={{ ...styles.metricValue, fontSize: "22px", color: customAnomalyCount > 0 ? "#dc3545" : "#28a745" }}>{customAnomalyCount}</div>
                </div>
                <div style={{ ...styles.metricCard, padding: "15px" }}>
                  <div style={styles.metricLabel}>Max Score</div>
                  <div style={{ ...styles.metricValue, fontSize: "22px" }}>{customMaxScore.toFixed(4)}</div>
                </div>
                <div style={{ ...styles.metricCard, padding: "15px" }}>
                  <div style={styles.metricLabel}>Current Status</div>
                  <div style={{ ...styles.metricValue, fontSize: "18px", color: customStatus === "HEALTHY" ? "#28a745" : customStatus === "CRITICAL_FAILURE" ? "#dc3545" : "#ffc107" }}>
                    {customStatus}
                  </div>
                </div>
              </div>

              {/* Progress Bar */}
              <div style={{ ...styles.card, padding: "15px", marginBottom: "20px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <span style={{ color: "#888", fontSize: "14px" }}>Simulation Progress</span>
                  <span style={{ color: "#00C9FF", fontWeight: "600" }}>{customTotalRecords > 0 ? ((customCurrentIndex / customTotalRecords) * 100).toFixed(1) : 0}%</span>
                </div>
                <div style={{ height: "10px", backgroundColor: "#333", borderRadius: "5px", overflow: "hidden" }}>
                  <div style={{
                    height: "100%",
                    width: `${customTotalRecords > 0 ? (customCurrentIndex / customTotalRecords) * 100 : 0}%`,
                    background: "linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%)",
                    transition: "width 0.1s ease"
                  }} />
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "20px" }}>
                {/* Controls */}
                <div style={styles.card}>
                  <h3>🎮 Simulation Controls</h3>

                  {/* Speed Control */}
                  <div style={{ marginBottom: "20px", padding: "15px", backgroundColor: "#1e1e2e", borderRadius: "8px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px" }}>
                      <span style={{ color: "#888" }}>⚡ Speed</span>
                      <span style={{ color: "#00C9FF" }}>{customSimSpeed}ms</span>
                    </div>
                    <input
                      type="range"
                      min="25"
                      max="500"
                      step="25"
                      value={customSimSpeed}
                      onChange={(e) => {
                        setCustomSimSpeed(Number(e.target.value));
                        customSpeedRef.current = Number(e.target.value);
                      }}
                      style={{ width: "100%", accentColor: "#00C9FF" }}
                    />
                  </div>

                  {/* Start/Stop Button */}
                  <button
                    onClick={async () => {
                      if (isCustomRunningRef.current) {
                        isCustomRunningRef.current = false;
                        setIsCustomSimulating(false);
                        return;
                      }

                      isCustomRunningRef.current = true;
                      setIsCustomSimulating(true);

                      await axios.post('http://127.0.0.1:8000/simulate-custom/reset');
                      setCustomCurrentIndex(0);
                      setCustomSimulationHistory([]);
                      setCustomAnomalyCount(0);
                      setCustomMaxScore(0);

                      const runNext = async () => {
                        if (!isCustomRunningRef.current) return;
                        try {
                          const response = await axios.get('http://127.0.0.1:8000/simulate-custom/next');
                          const data = response.data;

                          if (data.finished) {
                            isCustomRunningRef.current = false;
                            setIsCustomSimulating(false);
                            setCustomStatus("COMPLETE");
                            return;
                          }

                          setCustomScore(data.anomaly_score);
                          setCustomStatus(data.status);
                          setCustomCurrentIndex(data.index);
                          setCustomMaxScore(prev => Math.max(prev, data.anomaly_score));

                          if (data.status === "CRITICAL_FAILURE") {
                            setCustomAnomalyCount(prev => prev + 1);
                          }

                          setCustomSimulationHistory(prev => [...prev.slice(-49), {
                            time: data.index,
                            score: data.anomaly_score,
                            limit: data.threshold
                          }]);

                          if (isCustomRunningRef.current) {
                            setTimeout(runNext, customSpeedRef.current);
                          }
                        } catch (err) {
                          console.error("Simulation Error", err);
                          isCustomRunningRef.current = false;
                          setIsCustomSimulating(false);
                        }
                      };
                      runNext();
                    }}
                    style={{
                      padding: "16px 32px",
                      fontSize: "16px",
                      fontWeight: "600",
                      border: "none",
                      borderRadius: "12px",
                      cursor: "pointer",
                      background: isCustomSimulating
                        ? "linear-gradient(135deg, #dc3545 0%, #c82333 100%)"
                        : "linear-gradient(135deg, #28a745 0%, #218838 100%)",
                      color: "#fff",
                      width: "100%",
                      boxShadow: isCustomSimulating ? "0 4px 15px rgba(220, 53, 69, 0.4)" : "0 4px 15px rgba(40, 167, 69, 0.4)"
                    }}
                  >
                    {isCustomSimulating ? "⏹ STOP SIMULATION" : "▶ START LIVE SIMULATION"}
                  </button>

                  {/* Current Score Display */}
                  <div style={{ marginTop: "20px", padding: "20px", backgroundColor: "#1e1e2e", borderRadius: "8px", textAlign: "center" }}>
                    <div style={{ color: "#888", fontSize: "14px" }}>Current Anomaly Score</div>
                    <div style={{ fontSize: "36px", fontWeight: "bold", color: customScore > 0.0005 ? "#dc3545" : "#28a745" }}>
                      {customScore.toFixed(4)}
                    </div>
                  </div>

                  {/* Export to Excel Button */}
                  {customSimulationHistory.length > 0 && (
                    <button
                      onClick={async () => {
                        try {
                          const exportData = {
                            filename: batchFile?.name || "simulation",
                            results: customSimulationHistory.map((item, idx) => ({
                              index: item.time,
                              score: item.score,
                              is_anomaly: item.score > (modelInfo.threshold || 0.0005)
                            }))
                          };

                          const response = await axios.post(
                            'http://127.0.0.1:8000/export/batch-results',
                            exportData,
                            { responseType: 'blob' }
                          );

                          // Create download link
                          const url = window.URL.createObjectURL(new Blob([response.data]));
                          const link = document.createElement('a');
                          link.href = url;
                          link.setAttribute('download', `batch_analysis_${batchFile?.name?.replace('.csv', '') || 'results'}.xlsx`);
                          document.body.appendChild(link);
                          link.click();
                          link.remove();
                          window.URL.revokeObjectURL(url);
                        } catch (error) {
                          console.error("Export Error:", error);
                          alert("Failed to export: " + error.message);
                        }
                      }}
                      style={{
                        marginTop: "15px",
                        padding: "12px 24px",
                        fontSize: "14px",
                        fontWeight: "600",
                        border: "none",
                        borderRadius: "8px",
                        cursor: "pointer",
                        background: "linear-gradient(135deg, #17a2b8 0%, #138496 100%)",
                        color: "#fff",
                        width: "100%",
                        boxShadow: "0 4px 15px rgba(23, 162, 184, 0.3)"
                      }}
                    >
                      📥 Export to Excel
                    </button>
                  )}
                </div>

                {/* Live Chart */}
                <div style={{ ...styles.card, height: "350px" }}>
                  <h3>📈 Live Anomaly Detection</h3>
                  <ResponsiveContainer width="100%" height="85%">
                    <LineChart data={customSimulationHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="time" stroke="#666" />
                      <YAxis stroke="#666" />
                      <Tooltip contentStyle={{ backgroundColor: "#333", border: "1px solid #555" }} />
                      <ReferenceLine y={0.0005} label="Threshold" stroke="red" strokeDasharray="3 3" />
                      <Line type="monotone" dataKey="score" stroke="#00C9FF" strokeWidth={3} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ================= TAB 5: DIAGNOSTICS ================= */}
      {activeTab === "diagnostics" && (
        <div>
          <div style={styles.card}>
            <h2>🔍 Root Cause Analysis</h2>
            <p style={{ color: "#888" }}>Which sensor features contribute most to the model's decision?</p>
          </div>

          <div style={{ ...styles.card, height: "500px" }}>
            <h3>Feature Importance (Model Weights)</h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={featureImportance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis type="number" stroke="#888" />
                <YAxis dataKey="name" type="category" stroke="#888" width={80} />
                <Tooltip contentStyle={{ backgroundColor: "#262730", border: "1px solid #464b5c" }} />
                <Bar dataKey="importance" name="Importance">
                  {featureImportance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={`hsl(${200 + index * 5}, 70%, ${50 + index}%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ================= TAB 6: MODEL INFO ================= */}
      {activeTab === "modelinfo" && (
        <div>
          <div style={styles.grid3}>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Algorithm</div>
              <div style={styles.metricValue}>PCA</div>
              <div style={styles.metricDelta}>Unsupervised</div>
            </div>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Input Features</div>
              <div style={styles.metricValue}>20</div>
              <div style={styles.metricDelta}>4 Bearings × 5 Metrics</div>
            </div>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Training Sets</div>
              <div style={styles.metricValue}>3</div>
              <div style={styles.metricDelta}>NASA IMS</div>
            </div>
          </div>

          <div style={styles.card}>
            <h3>📋 Model Configuration</h3>
            <table style={styles.table}>
              <tbody>
                <tr><td style={styles.td}><strong>Model Type</strong></td><td style={styles.td}>Principal Component Analysis (Dimensionality Reduction)</td></tr>
                <tr><td style={styles.td}><strong>Training Data Source</strong></td><td style={styles.td}>NASA IMS Bearing Datasets (Sets 1, 2, 3)</td></tr>
                <tr><td style={styles.td}><strong>Feature Scaling</strong></td><td style={styles.td}>StandardScaler (mean=0, std=1)</td></tr>
                <tr><td style={styles.td}><strong>Anomaly Detection</strong></td><td style={styles.td}>Euclidean distance between original & reconstructed</td></tr>
                <tr><td style={styles.td}><strong>Anomaly Threshold</strong></td><td style={styles.td}>{modelInfo.threshold.toFixed(6)}</td></tr>
              </tbody>
            </table>
          </div>

          <h3>📊 Feature Breakdown</h3>
          <div style={{ ...styles.grid2, gridTemplateColumns: "repeat(4, 1fr)" }}>
            <div style={styles.card}>
              <h4>🔴 Bearing 1</h4>
              <p style={{ color: "#888", fontSize: "12px" }}>Primary load-bearing component. Most sensitive to radial loads.</p>
            </div>
            <div style={styles.card}>
              <h4>🟡 Bearing 2</h4>
              <p style={{ color: "#888", fontSize: "12px" }}>Support bearing. Detects shaft misalignment issues.</p>
            </div>
            <div style={styles.card}>
              <h4>🟢 Bearing 3</h4>
              <p style={{ color: "#888", fontSize: "12px" }}>Secondary load-bearing. Often fails after primary bearing.</p>
            </div>
            <div style={styles.card}>
              <h4>🔵 Bearing 4</h4>
              <p style={{ color: "#888", fontSize: "12px" }}>Outboard bearing. Earliest indicator of external shock loads.</p>
            </div>
          </div>

          <div style={styles.grid2}>
            <div style={styles.card}>
              <h4>RMS (Root Mean Square)</h4>
              <p style={{ color: "#888" }}>Represents total energy in the signal. Higher values = more vibration.</p>
              <h4>Kurtosis</h4>
              <p style={{ color: "#888" }}>Measures "peakedness" of distribution. Normal: ~3, Impulsive: &gt;5. Early failure indicator.</p>
            </div>
            <div style={styles.card}>
              <h4>Skewness</h4>
              <p style={{ color: "#888" }}>Asymmetry of vibration pattern. Healthy: ~0, Degrading: shifts from 0.</p>
              <h4>Crest Factor</h4>
              <p style={{ color: "#888" }}>Peak/RMS ratio. Healthy bearings: 3-4. Failing bearings: &gt;10.</p>
            </div>
          </div>

          <div style={styles.card}>
            <h3>✅ When to Take Action</h3>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Anomaly Score</th>
                  <th style={styles.th}>Status</th>
                  <th style={styles.th}>Action</th>
                </tr>
              </thead>
              <tbody>
                <tr><td style={styles.td}>0.0 - 0.0001</td><td style={{ ...styles.td, color: "#92FE9D" }}>✅ Healthy</td><td style={styles.td}>Continue normal operation</td></tr>
                <tr><td style={styles.td}>0.0001 - 0.0005</td><td style={{ ...styles.td, color: "#FFD700" }}>🟡 Monitor</td><td style={styles.td}>Track trends, schedule inspection in 4-6 weeks</td></tr>
                <tr><td style={styles.td}>0.0005 - threshold</td><td style={{ ...styles.td, color: "#FFA500" }}>🟠 Alert</td><td style={styles.td}>Schedule maintenance within 1-2 weeks</td></tr>
                <tr><td style={styles.td}>&gt; threshold</td><td style={{ ...styles.td, color: "#ff4b4b" }}>🚨 Critical</td><td style={styles.td}>Stop machine immediately, replace bearing</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;