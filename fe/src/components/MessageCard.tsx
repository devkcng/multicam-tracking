import { Card, CardContent, Typography } from "@mui/joy";

type MessageCardProps = {
  content: string;
  isUser: boolean;
  timestamp?: string; // Cho phép truyền thời gian từ bên ngoài (optional)
};

const MessageCard = ({ content, isUser, timestamp }: MessageCardProps) => {
  const time = timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <Card
      component="li"
      sx={{
        padding: 0,
        minWidth: 100,
        border: "none",
        display: "flex",
        alignSelf: isUser ? "flex-end" : "flex-start",
      }}
    >
      <CardContent
        sx={{
          background: isUser ? "#1976d2" : "#333",
          color: "#fff",
          display: "inline-block",
          borderRadius: "12px",
          paddingX: 2,
          paddingY: 1,
          maxWidth: "70%",
        }}
      >
        <Typography level="body-md" sx={{ mb: 0.5, color: "#fff" }}>
          {content}
        </Typography>
        <Typography level="body-sm" sx={{ fontSize: 10, textAlign: "right", opacity: 0.6 }}>
          {time}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default MessageCard;
