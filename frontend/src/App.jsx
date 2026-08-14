import { useState } from "react";
import "./App.css";

import NavBar from "./components/NavBar";
import FileUpload from "./components/FileUpload";
import ChatBox from "./components/ChatBox";
import Answer from "./components/Answer";
import Login from "./components/Login";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem("token"));

  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsLoggedIn(false);
    setFile(null);
    setMessages([]);
  };

  // Show Login page if user is not authenticated
  if (!isLoggedIn) {
    return <Login onLogin={handleLogin} />;
  }

  // Show the RAG application after login
  return (
    <div className="app">
      <div className="sidebar">
        <NavBar />

        <button onClick={handleLogout}>Logout</button>

        <FileUpload file={file} setFile={setFile} />

        {file && (
          <div className="file-card">
            <span>📄 {file.name}</span>

            <button className="remove-btn" onClick={() => setFile(null)}>
              ✖
            </button>
          </div>
        )}
      </div>

      <div className="main-content">
        <Answer messages={messages} />

        <ChatBox messages={messages} setMessages={setMessages} />
      </div>
    </div>
  );
}

export default App;
