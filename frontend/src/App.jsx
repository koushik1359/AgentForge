import { useState } from "react";
import "./App.css";

function App() {
  const [userRequest, setUserRequest] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generateAgent = async () => {
    if (!userRequest.trim()) {
      setError("Please enter an agent idea.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/agents/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_request: userRequest,
        }),
      });

      if (!response.ok) {
        throw new Error("Backend request failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Could not connect to backend. Make sure backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h1>AgentForge</h1>
        <p className="subtitle">
          Generate specialized AI agent blueprints from a simple user goal.
        </p>

        <textarea
          value={userRequest}
          onChange={(e) => setUserRequest(e.target.value)}
          placeholder="Example: Create an agent that sends me daily AI news alerts"
        />

        <button onClick={generateAgent} disabled={loading}>
          {loading ? "Generating..." : "Generate Agent"}
        </button>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">
            <h2>Generated Agent Blueprint</h2>

            <h3>Goal</h3>
            <p>{result.goal}</p>

            <h3>Frequency</h3>
            <p>{result.frequency}</p>

            <h3>Output Type</h3>
            <p>{result.output_type}</p>

            <h3>Tools Needed</h3>
            <ul>
              {result.tools_needed.map((tool, index) => (
                <li key={index}>{tool}</li>
              ))}
            </ul>

            <h3>System Prompt</h3>
            <pre>{result.system_prompt}</pre>

            <h3>Workflow Steps</h3>
            <pre>{JSON.stringify(result.workflow_steps, null, 2)}</pre>

            <h3>Memory Configuration</h3>
            <pre>{JSON.stringify(result.memory_config, null, 2)}</pre>

            <h3>Validation</h3>
            <pre>{JSON.stringify(result.validation_result, null, 2)}</pre>

            <h3>Mode</h3>
            <p>{result.mode === "true" ? "Mock Mode" : result.mode}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;