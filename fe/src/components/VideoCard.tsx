import {
  Card,
  CardContent,
  CardCover,
  Typography,
  Slider,
  Box,
} from "@mui/joy";
import React, { useRef, useState, useEffect } from "react";

type VideoCardProps = {
  videoSrc: string;
  title: string;
};

const VideoCard = ({ videoSrc, title }: VideoCardProps) => {
  const videoRef = useRef<HTMLVideoElement>(null); // Ref để truy cập video element
  const [currentTime, setCurrentTime] = useState(0); // Thời gian hiện tại của video
  const [duration, setDuration] = useState(0); // Thời lượng video

  // Cập nhật thời gian hiện tại khi video phát
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateTime = () => setCurrentTime(video.currentTime);
    const setVideoDuration = () => setDuration(video.duration);

    video.addEventListener("timeupdate", updateTime);
    video.addEventListener("loadedmetadata", setVideoDuration);

    // Dọn dẹp event listeners khi component unmount
    return () => {
      video.removeEventListener("timeupdate", updateTime);
      video.removeEventListener("loadedmetadata", setVideoDuration);
    };
  }, []);

  // Xử lý khi người dùng tua thời gian
  const handleSeekChange = (event: Event, newValue: number | number[]) => {
    const video = videoRef.current;
    if (video && typeof newValue === "number") {
      video.currentTime = newValue; // Cập nhật thời gian video
      setCurrentTime(newValue); // Đồng bộ state
    }
  };

  // Hàm format thời gian từ giây sang phút:giây
  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes.toString().padStart(2, "0")}:${seconds
      .toString()
      .padStart(2, "0")}`;
  };

  return (
    <Card component="li" sx={{ minWidth: 300, border: "none" }}>
      <CardCover>
        <video
          ref={videoRef}
          style={{
            borderRadius: "8px",
            width: "100%",
            height: "100%",
          }}
          autoPlay
          loop
          muted
        >
          <source src={videoSrc || "/video/demo2.mp4"} type="video/mp4" />
        </video>
      </CardCover>
      <CardContent
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Typography
          level="body-lg"
          textColor="#fff"
          sx={{
            background: "black",
            borderRadius: "8px",
            width: "fit-content",
            paddingX: 1,
            textAlign: "center",
            fontWeight: "lg",
            fontSize: "xs",
            mt: { xs: 12, sm: 18 },
          }}
        >
          {title || "cam1"}
        </Typography>

        {/* Thanh tua thời gian và đếm thời gian */}
        <Box
          sx={{
            width: "80%",
            mt: 2,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 1,
          }}
        >
          <Slider
            value={currentTime}
            min={0}
            max={duration || 100} // Nếu duration chưa load, dùng giá trị mặc định
            onChange={handleSeekChange}
            sx={{
              color: "#fff",
              padding: "0px",
            }}
          />
          <Typography level="body-sm" textColor="#fff">
            {formatTime(currentTime)} / {formatTime(duration)}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default VideoCard;
