import { useState } from "react";
import axios from "axios";
import "./Login.css";

const API = "http://localhost:8080";

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await axios.post(`${API}/auth/login`, {
        username,
        password,
      });

      console.log("Login response:", response.data);

      // Save JWT token
      localStorage.setItem("token", response.data);

      // Tell App that login was successful
      onLogin();
    } catch (error) {
      console.error("Login error:", error);

      if (error.response) {
        setError("Invalid username or password.");
      } else {
        setError("Unable to connect to server.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        {/* Logo / Branding */}
        <div className="login-logo">
          <div className="logo-icon">📄</div>
          <h1>PDF Chat</h1>
        </div>

        <h2>Welcome back</h2>
        <p className="login-subtitle">
          Login to continue chatting with your documents
        </p>

        <form onSubmit={handleLogin}>
          {/* Username */}
          <div className="input-group">
            <label htmlFor="username">Username</label>

            <div className="input-wrapper">
              <span className="input-icon">👤</span>

              <input
                id="username"
                type="text"
                placeholder="Enter your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                required
              />
            </div>
          </div>

          {/* Password */}
          <div className="input-group">
            <label htmlFor="password">Password</label>

            <div className="input-wrapper">
              <span className="input-icon">🔒</span>

              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                required
              />
            </div>
          </div>

          {/* Error */}
          {error && <div className="login-error">⚠️ {error}</div>}

          {/* Login button */}
          <button type="submit" className="login-button" disabled={loading}>
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <p className="login-footer">
          Ask questions. Get answers. Understand your PDFs.
        </p>
      </div>
    </div>
  );
}

export default Login;
