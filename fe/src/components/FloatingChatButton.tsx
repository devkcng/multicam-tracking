import { Box, Button } from "@mui/joy";
import { useState } from "react";
import ChatBox from "./ChatBox";
import ChatBubbleIcon from "@mui/icons-material/ChatBubble"; // nếu có icon

const FloatingChatButton = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Chat popup */}
        <Box
          sx={{
            position: "fixed",
            bottom: 80,
            right: 24,
            width: 400,
            height: 500,
            borderRadius: "md",
            boxShadow: "lg",
            zIndex: 1300,
            backgroundColor: "#fff",
            display: isOpen ? "display":"none"
          }}
        >
          <ChatBox />
        </Box>

      {/* Floating button */}
      <Box
        sx={{
          position: "fixed",
          bottom: 16,
          right: 16,
          zIndex: 1200,
        }}
      >
        <Button
          color="primary"
          variant="solid"
          size="lg"
          sx={{ borderRadius: "50%", width: 56, height: 56 }}
          onClick={() => setIsOpen((prev) => !prev)}
        >
          <ChatBubbleIcon />
        </Button>
      </Box>
    </>
  );
};

export default FloatingChatButton;
