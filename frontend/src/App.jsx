import { useState } from "react";
import "./App.css";

import NavBar from "./components/NavBar";
import FileUpload from "./components/FileUpload";
import ChatBox from "./components/ChatBox";
import Answer from "./components/Answer";

function App() {
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);
  return (
    <div className="app">
      <div className="sidebar">
        <NavBar />

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
