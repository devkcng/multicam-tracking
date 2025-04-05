import { Card, CardCover, Typography, Slider, Box, IconButton } from "@mui/joy";
import React, { useRef, useState, useEffect } from "react";
import CloseIcon from "@mui/icons-material/Close";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import VolumeOffIcon from "@mui/icons-material/VolumeOff";

export type VideoCardProps = {
  videoSrc: string;
  title: string;
  isSelected?: boolean;
  onClick?: () => void;
  onClose?: () => void;
  autoPlay?: boolean;
  startTime?: number;
  loop?: boolean;
  muted?: boolean;
};

const VideoCard = ({
  videoSrc,
  title,
  isSelected = false,
  onClick,
  onClose,
  autoPlay = true,
  startTime = 0,
  loop = false,
  muted = true,
}: VideoCardProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(startTime);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [isMuted, setIsMuted] = useState(muted);
  const [showControls, setShowControls] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // Set initial time and play state
    video.currentTime = startTime;
    if (autoPlay) {
      video.play().catch((error) => {
        console.error("Autoplay failed:", error);
        setIsPlaying(false);
      });
    }

    const updateTime = () => setCurrentTime(video.currentTime);
    const setVideoDuration = () => setDuration(video.duration);

    video.addEventListener("timeupdate", updateTime);
    video.addEventListener("loadedmetadata", setVideoDuration);

    return () => {
      video.removeEventListener("timeupdate", updateTime);
      video.removeEventListener("loadedmetadata", setVideoDuration);
    };
  }, [startTime, loop, isPlaying, isMuted]);

  const handleSeekChange = (event: Event, newValue: number | number[]) => {
    const video = videoRef.current;
    if (video && typeof newValue === "number") {
      video.currentTime = newValue;
      setCurrentTime(newValue);

      // If video was paused, play it after seek if autoPlay is true
      if (autoPlay && video.paused) {
        video.play().catch((error) => {
          console.error("Play after seek failed:", error);
        });
      }
    }
  };

  const togglePlayPause = () => {
    const video = videoRef.current;
    if (video) {
      if (video.paused) {
        video.play().then(() => setIsPlaying(true));
      } else {
        video.pause();
        setIsPlaying(false);
      }
    }
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (video) {
      video.muted = !video.muted;
      setIsMuted(video.muted);
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
    e.stopPropagation();
    if (onClose) onClose();
  };

  const handleCardClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!isSelected && onClick) {
      onClick();
    } else {
      togglePlayPause();
    }
  };

  return (
    <Card
      component="li"
      onClick={handleCardClick}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
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
          autoPlay={autoPlay}
          loop={loop}
          muted={isMuted}
        >
          <source src={videoSrc || "/video/demo2.mp4"} type="video/mp4" />
        </video>
      </CardCover>

      {/* Play/Pause Button (center) */}
      {!isPlaying && (
        <IconButton
          onClick={(e) => {
            e.stopPropagation();
            togglePlayPause();
          }}
          sx={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            color: "#fff",
            background: "rgba(0, 0, 0, 0.5)",
            borderRadius: "50%",
            zIndex: 1,
            "&:hover": {
              background: "rgba(0, 0, 0, 0.7)",
            },
            opacity: showControls || !isPlaying ? 1 : 0,
            transition: "opacity 0.3s ease-in-out",
          }}
        >
          <PlayArrowIcon fontSize="large" />
        </IconButton>
      )}

      {/* Title */}
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

      {/* Controls Bar */}
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
          background: "rgba(0, 0, 0, 0.5)",
          borderRadius: "8px",
          padding: 1,
          zIndex: 1,
          transition: "all 0.5s ease-in-out",
          opacity: showControls || isSelected ? 1 : 0,
        }}
      >
        <Box
          sx={{ width: "100%", display: "flex", alignItems: "center", gap: 1 }}
        >
          <IconButton
            onClick={(e) => {
              e.stopPropagation();
              togglePlayPause();
            }}
            size="sm"
            sx={{ color: "#fff" }}
          >
            {isPlaying ? (
              <PauseIcon sx={{ color: "white" }} />
            ) : (
              <PlayArrowIcon sx={{ color: "white" }} />
            )}
          </IconButton>

          <Slider
            value={currentTime}
            min={0}
            max={duration || 100}
            onChange={handleSeekChange}
            sx={{
              flex: 1,
              color: "#fff",
              padding: "0px",
            }}
          />

          <IconButton
            onClick={(e) => {
              e.stopPropagation();
              toggleMute();
            }}
            size="sm"
            sx={{ color: "#fff" }}
          >
            {isMuted ? (
              <VolumeOffIcon sx={{ color: "white" }} />
            ) : (
              <VolumeUpIcon />
            )}
          </IconButton>

          <Typography
            level="body-sm"
            textColor="#fff"
            sx={{
              background: "black",
              borderRadius: "8px",
              paddingX: 1,
              fontSize: "xs",
              minWidth: "80px",
              textAlign: "center",
            }}
          >
            {formatTime(currentTime)} / {formatTime(duration)}
          </Typography>
        </Box>
      </Box>

      {/* Close Button (only shown when video is selected) */}
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
