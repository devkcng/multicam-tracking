import { Box, Typography } from "@mui/joy";
import SearchBar from "./components/SearchBar";
import VideoCard from "./components/VideoCard";
import ObjectItem from "./components/ObjectItem";
import TrackingSection from "./components/TrackingSection";
import FloatingChatButton from "./components/FloatingChatButton";
import { VideoCardProps } from "./components/VideoCard";
import { ObjectProps } from "./components/ObjectItem";
import { useState, useEffect } from "react";

function App() {
  const [videos, setVideos] = useState<VideoCardProps[]>([]);

  const [people_ids, setPeopleIds] = useState<ObjectProps[]>([]);

  // Run load people ids whenever enter web
  useEffect(() => {
    const fetchIds = async () => {
      const ids = await loadPersonIds();
      setPeopleIds(ids);
    };
    fetchIds();
  }, []);

  // First load ids of people in videos
  const loadPersonIds = async (): Promise<ObjectProps[]> => {
    try {
      const response = await fetch("https://engaged-hagfish-usefully.ngrok-free.app/all_person_ids", {
        method: "GET",
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const result = data.all_ids.map(id => ({ id }));

      return result;

    } catch (error) {
      console.error("Error fetching person IDs:", error);
      return [];
    }
  };

  // Handle filter submission event
  const handleFilter = async (selectedItem: string) => {
    return;

    console.log(selectedItem);
    try {
      const response = await fetch("https://above-ruling-ringtail.ngrok-free.app/draw-matching", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({
          person_id: parseInt(selectedItem),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const videoData = data.videos.map((item: any) => ({
        title: item.cam_id,
        videoSrc: item.url._url,
      }));

      setVideos(videoData);

    } catch (error) {
      console.error("Error while filtering:", error);
    }
  }

  return (
    <>
      <Box
        sx={{
          width: "auto",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 2,
          padding: 2,
        }}
      >
        <Typography level="h1" sx={{ textAlign: "center" }}>
          Tracking Assistant
        </Typography>

        <Box
          sx={{
            display: "flex",
            flexDirection: "row",
            gap: 2,
            alignItems: "center",
          }}
        >
          {/* Video section */}
          <Box
            sx={{
              width: "80vw",
              height: "auto",
              border: "1px solid black",
              borderRadius: "lg",
              display: "flex",
              flexDirection: "column",
              gap: 2,
              padding: 1,
              alignItems: "center",
            }}
          >
            <SearchBar onSearch={() => { }} initialValue=""></SearchBar>
            <TrackingSection videos={videos}></TrackingSection>
          </Box>
          {/* Filter section */}
          <Box
            sx={{
              width: "auto",
              height: "auto",
              padding: 2,
            }}
          >
            <ObjectItem items={people_ids} onClick={handleFilter}></ObjectItem>
          </Box>
        </Box>
      </Box>
      <FloatingChatButton />
    </>
  );
}

export default App;
