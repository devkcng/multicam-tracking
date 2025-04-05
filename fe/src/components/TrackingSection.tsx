import { Box } from "@mui/joy";
import React, { useState } from "react";
import VideoCard, { VideoCardProps } from "./VideoCard";

type TrackingSectionProps = {
  videos: VideoCardProps[];
};

const TrackingSection = ({
  videos = [
    {
      videoSrc: "/video/demo2.mp4",
      title: "cam1",
    },
    {
      videoSrc: "/video/demo2.mp4",
      title: "cam2",
    },
    {
      videoSrc: "/video/demo2.mp4",
      title: "cam3",
    },
    {
      videoSrc: "/video/demo2.mp4",
      title: "cam4",
    },
    {
      videoSrc: "/video/demo2.mp4",
      title: "cam5",
    },
  ],
}: TrackingSectionProps) => {
  const [selectedVideoIndex, setSelectedVideoIndex] = useState<number | null>(
    null
  );

  const handleVideoClick = (index: number) => {
    setSelectedVideoIndex(selectedVideoIndex === index ? null : index); // Toggle chọn/hủy chọn
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexWrap: "wrap",
        gap: 2,
        padding: 2,
        justifyContent: "center",
        position: "relative",
        minHeight: "100vh", // Đảm bảo đủ không gian khi phóng to
      }}
    >
      {videos.map((video, index) => (
        <Box
          key={index}
          sx={{
            display:
              selectedVideoIndex === null || selectedVideoIndex === index
                ? "block"
                : "none", // Ẩn các video không được chọn
          }}
        >
          <VideoCard
            title={video.title}
            videoSrc={video.videoSrc}
            isSelected={selectedVideoIndex === index} // Truyền trạng thái được chọn
            onClick={() => handleVideoClick(index)} // Xử lý click
          />
        </Box>
      ))}
    </Box>
  );
};

export default TrackingSection;
