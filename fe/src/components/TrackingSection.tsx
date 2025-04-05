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
    setSelectedVideoIndex(index); // Chỉ phóng to, không toggle
  };

  const handleVideoClose = () => {
    setSelectedVideoIndex(null); // Thu nhỏ lại
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
        minHeight: "100vh",
      }}
    >
      {videos.map((video, index) => (
        <Box
          key={index}
          sx={{
            display:
              selectedVideoIndex === null || selectedVideoIndex === index
                ? "block"
                : "none",
          }}
        >
          <VideoCard
            title={video.title}
            videoSrc={video.videoSrc}
            isSelected={selectedVideoIndex === index}
            onClick={() => handleVideoClick(index)} // Xử lý phóng to
            onClose={handleVideoClose} // Xử lý thu nhỏ
          />
        </Box>
      ))}
    </Box>
  );
};

export default TrackingSection;
