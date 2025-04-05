import { Card, CardCover, Typography, Slider, Box, IconButton } from "@mui/joy";
import React, { useRef, useState, useEffect } from "react";
import CloseIcon from "@mui/icons-material/Close";

export type VideoCardProps = {
  videoSrc: string;
  title: string;
  isSelected?: boolean;
  onClick?: () => void;
  onClose?: () => void; // Thêm prop để xử lý thu nhỏ
};

const VideoCard = ({
  videoSrc,
  title,
  isSelected = false,
  onClick,
  onClose,
}: VideoCardProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateTime = () => setCurrentTime(video.currentTime);
    const setVideoDuration = () => setDuration(video.duration);

    video.addEventListener("timeupdate", updateTime);
    video.addEventListener("loadedmetadata", setVideoDuration);

    return () => {
      video.removeEventListener("timeupdate", updateTime);
      video.removeEventListener("loadedmetadata", setVideoDuration);
    };
  }, []);

  const handleSeekChange = (event: Event, newValue: number | number[]) => {
    const video = videoRef.current;
    if (video && typeof newValue === "number") {
      video.currentTime = newValue;
      setCurrentTime(newValue);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes.toString().padStart(2, "0")}:${seconds
      .toString()
      .padStart(2, "0")}`;
  };

  const handleCloseClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Ngăn click lan lên Card
    if (onClose) onClose(); // Gọi onClose để thu nhỏ
  };

  const handleCardClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!isSelected && onClick) onClick(); // Chỉ phóng to khi chưa được chọn
  };

  return (
    <Card
      component="li"
      onClick={handleCardClick} // Chỉ xử lý phóng to
      sx={{
        minWidth: isSelected ? "80vw" : 300,
        minHeight: isSelected ? "70vh" : 200,
        border: "none",
        cursor: "pointer",
        position: "relative",
        background: "transparent",
        transition:
          "min-width 0.5s ease-in-out, min-height 0.5s ease-in-out, transform 0.3s ease-in-out",
        transform: isSelected ? "scale(1)" : "scale(1)",
      }}
    >
      <CardCover>
        <video
          ref={videoRef}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            transition: "all 0.5s ease-in-out",
          }}
          autoPlay
          loop
          muted
        >
          <source src={videoSrc || "/video/demo2.mp4"} type="video/mp4" />
        </video>
      </CardCover>

      {/* Title absolute */}
      <Typography
        level="body-lg"
        textColor="#fff"
        sx={{
          position: "absolute",
          bottom: -20,
          left: "50%",
          transform: "translateX(-50%)",
          background: "black",
          borderRadius: "8px",
          width: "fit-content",
          paddingX: 1,
          textAlign: "center",
          fontWeight: "lg",
          fontSize: "xs",
          zIndex: 1,
          transition: "all 0.5s ease-in-out",
        }}
      >
        {title || "cam1"}
      </Typography>

      {/* Thanh tua thời gian và đếm thời gian */}
      <Box
        sx={{
          position: "absolute",
          bottom: 8,
          left: "50%",
          transform: "translateX(-50%)",
          width: "70%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 1,
          background: "rgba(0, 0, 0, 0.2)",
          borderRadius: "8px",
          padding: 1,
          zIndex: 1,
          transition: "all 0.5s ease-in-out",
          opacity: isSelected ? 1 : 0.8,
        }}
      >
        <Slider
          value={currentTime}
          min={0}
          max={duration || 100}
          onChange={handleSeekChange}
          sx={{
            color: "#fff",
            padding: "0px",
          }}
        />
        <Typography
          level="body-sm"
          textColor="#fff"
          sx={{
            background: "black",
            borderRadius: "8px",
            paddingX: 1,
            fontSize: "xs",
          }}
        >
          {formatTime(currentTime)} / {formatTime(duration)}
        </Typography>
      </Box>

      {/* Nút Close (chỉ hiển thị khi video được chọn) */}
      {isSelected && (
        <IconButton
          onClick={handleCloseClick}
          sx={{
            position: "absolute",
            top: -20,
            right: 120,
            color: "#fff",
            background: "rgba(231, 21, 21, 0.5)",
            borderRadius: "50%",
            zIndex: 2,
            "&:hover": {
              background: "rgba(231, 21, 21, 0.7)",
            },
            transition: "opacity 0.3s ease-in-out",
            opacity: 1,
          }}
        >
          <CloseIcon sx={{ color: "white" }} />
        </IconButton>
      )}
    </Card>
  );
};

export default VideoCard;
