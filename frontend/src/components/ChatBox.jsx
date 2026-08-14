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

    // Clear input
    setQuestion("");

    try {
      console.log("Sending question:", userQuestion);

      // Get JWT token
      const token = localStorage.getItem("token");

      if (!token) {
        throw new Error("No authentication token found");
      }

      // Send question to Spring Boot backend
      const response = await axios.post(
        "http://localhost:8080/query",
        {
          question: userQuestion,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );

      console.log("Response:", response.data);

      // Show bot response
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: response.data.answer,
        },
      ]);
    } catch (error) {
      console.error("AXIOS ERROR:", error);

      if (error.response) {
        console.error("Status:", error.response.status);
        console.error("Response:", error.response.data);
      }

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
