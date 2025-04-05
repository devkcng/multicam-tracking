import { useState, useRef, useEffect } from "react";
import { Box, Input, Button, CircularProgress } from "@mui/joy";
import Message from "./MessageCard";


type MessageType = {
  id: number;
  text: string;
  sender: "user" | "assistant";
};

const ChatBox = () => {
  const [messages, setMessages] = useState<MessageType[]>([{
    id: 1, 
    text: "Hi! How can I help you?", 
    sender: "assistant"
  }]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (inputValue.trim() === "") return;

    const newMessage: MessageType = {
      id: Date.now(),
      text: inputValue.trim(),
      sender: "user",
    };

    setMessages((prev) => [...prev, newMessage]);
    setInputValue("");
    setLoading(true);

    try {
      const response = await fetch("https://above-ruling-ringtail.ngrok-free.app/", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
      });
  
      const data = await response.json();
  
      const assistantMessage: MessageType = {
        id: Date.now() + 1,
        text: data.message || "No response from server",
        sender: "assistant",
      };
      
      // Add new message from server
      setMessages((prev) => [...prev, assistantMessage]);
  
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 2,
          text: "Internal server error!",
          sender: "assistant",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        height: "100%",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        border: "1px solid #ccc",
        borderRadius: "md",
        background: "#f5f5f5",
        boxShadow: "lg",
      }}
    >
      {/* Message list */}
      <Box
        sx={{
          flexGrow: 1, // lấy hết không gian còn lại
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 1,
          padding: 2,
        }}
      >
        {messages.map((msg) => (
          <Message
            key={msg.id}
            content={msg.text}
            isUser={msg.sender === "user"}
          />
        ))}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input + send */}
      <Box sx={{ display: "flex", gap: 1, p: 1, borderTop: "1px solid #ddd" }}>
        <Input
          placeholder="What do you think?"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => {
            if (!loading && e.key === "Enter") handleSend();
          }}
          fullWidth
        />
        <Button onClick={handleSend} disabled={loading}>Send</Button>
      </Box>

      {loading && (
        <Box sx={{ display: "flex", justifyContent: "center", padding: 2 }}>
          <CircularProgress />
        </Box>
      )}

    </Box>
  );
};

export default ChatBox;