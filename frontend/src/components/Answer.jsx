function Answer({ messages }) {
  return (
    <div className="chat-window">
      {messages.map((message, index) => (
        <div
          key={index}
          className={message.sender === "user" ? "user-message" : "bot-message"}
        >
          {message.text}
        </div>
      ))}
    </div>
  );
}

export default Answer;
