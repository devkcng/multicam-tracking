import { Box } from "@mui/joy";
import React, { useState } from "react";
import VideoCard, { VideoCardProps } from "./VideoCard";

type TrackingSectionProps = {
  videos: VideoCardProps[];
};

const TrackingSection = ({ videos }: TrackingSectionProps) => {
  const [selectedVideoIndex, setSelectedVideoIndex] = useState<number | null>(
    null
  );

  const handleVideoClick = (index: number) => {
    setSelectedVideoIndex(index);
  };

  const handleVideoClose = () => {
    setSelectedVideoIndex(null);
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexWrap: "wrap",
        gap: 1,
        justifyContent: "center",
        position: "relative",
        minHeight: "76vh",
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
            {...video} // Truyền toàn bộ props từ video
            isSelected={selectedVideoIndex === index} // Ghi đè isSelected
            onClick={() => handleVideoClick(index)} // Ghi đè onClick
            onClose={handleVideoClose} // Ghi đè onClose
          />
        </Box>
      ))}
    </Box>
  );
};

export default TrackingSection;
