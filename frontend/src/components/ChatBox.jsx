import { useState } from "react";
import axios from "axios";

function ChatBox({ messages, setMessages }) {
  const [question, setQuestion] = useState("");

  const handleSend = async () => {
    // Don't send empty questions
    if (!question.trim()) return;

    const userQuestion = question;

    // Show user's message immediately
    setMessages((prev) => [
      ...prev,
      {
        sender: "user",
        text: userQuestion,
      },
    ]);

    // Clear the input box
    setQuestion("");

    try {
      console.log("Sending question:", userQuestion);

      // Send question to Node.js backend
      const response = await axios.post("http://localhost:8080/query", {
        question: userQuestion,
      });
      console.log("Response:", response.data);

      // Show bot's response
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: response.data.answer,
        },
      ]);
    } catch (error) {
      console.error("AXIOS ERROR:", error);

      // Show error message if server fails
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "Server error. Please try again.",
        },
      ]);
    }
  };

  return (
    <div className="chat-input-container">
      <input
        type="text"
        placeholder="Ask about your PDF..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            handleSend();
          }
        }}
      />

      <button onClick={handleSend}>➤</button>
    </div>
  );
}

export default ChatBox;
