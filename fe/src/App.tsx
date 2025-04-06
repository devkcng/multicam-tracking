import { Box, Typography } from "@mui/joy";
import { useEffect, useState } from "react";
import axios from "axios"; // Import axios
import FloatingChatButton from "./components/FloatingChatButton";
import ObjectItem, { ObjectProps } from "./components/ObjectItem"; // Import ObjectProps
import SearchBar from "./components/SearchBar";
import TrackingSection from "./components/TrackingSection";
import apiClient from "./config";
import { VideoCardProps } from "./components/VideoCard";

function App() {
  const [videos, setVideos] = useState<VideoCardProps[]>([]);
  const [originalVideos, setOriginalVideos] = useState<VideoCardProps[]>([]);
  const [items, setItems] = useState<ObjectProps[]>([]); // State cho items của ObjectItem
  const [error, setError] = useState<string | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Fetch videos
  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const rootResponse = await apiClient.get("/camera-ids");
        console.log("API data:", rootResponse.data);

        const videoData = rootResponse.data.videos.map((item: any) => ({
          title: item.cam_id,
          videoSrc: item.url._url,
          autoPlay: true,
          startTime: 0,
          loop: false,
          muted: true,
          isSelected: false,
        }));

        setVideos(videoData);
        setOriginalVideos(videoData);
      } catch (error: any) {
        setError(error.message || "Failed to fetch data");
        console.error("Error fetching data:", error);
      }
    };
    fetchVideos();
  }, []);

  // Fetch person IDs for ObjectItem
  useEffect(() => {
    const fetchPersonIds = async () => {
      try {
        const response = await apiClient.get("all_person_ids");
        console.log("Person IDs data:", response.data);

        // Chuyển đổi dữ liệu từ all_ids thành mảng ObjectProps
        const personItems = response.data.all_ids.map((id: string) => ({
          id, // Tạo object với thuộc tính id
        }));

        setItems(personItems);
      } catch (error: any) {
        setError(error.message || "Failed to fetch person IDs");
        console.error("Error fetching person IDs:", error);
      }
    };
    fetchPersonIds();
  }, []);

  const handleSearch = async (searchTerm: string) => {
    if (!searchTerm.trim()) {
      setVideos(originalVideos);
      return;
    }

    setSearchLoading(true);
    setSearchError(null);

    try {
      const response = await apiClient.post("/search/", {
        text: searchTerm,
      });
      console.log("check data: ", response.data[2].time_sec);
      console.log("check data: ", response.data[2].video_url);
      const searchVideoData = response.data.map((item: any) => ({
        title: item.camera_id,
        videoSrc: item.video_url,
        autoPlay: false,
        startTime: item.time_sec,
        loop: false,
        muted: true,
      }));
      console.log("search video data: ", searchVideoData[2]);
      setVideos(searchVideoData);
    } catch (error: any) {
      setSearchError(error.message || "Failed to perform search");
      console.error("Search error:", error);
    } finally {
      setSearchLoading(false);
    }
  };

  // Hàm xử lý khi chọn một person ID từ ObjectItem
  const handleObjectSelect = (selectedId: string) => {
    console.log("Selected person ID:", selectedId);
    // Bạn có thể thêm logic để xử lý selectedId, ví dụ: lọc videos dựa trên ID
  };

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
          Surveillance Assistant
        </Typography>

        <Box
          sx={{
            display: "flex",
            flexDirection: "row",
            gap: 2,
            alignItems: "center",
          }}
        >
          <Box
            sx={{
              width: "80vw",
              height: "auto",
              maxHeight: "80vh",
              border: "1px solid black",
              borderRadius: "lg",
              display: "flex",
              flexDirection: "column",
              gap: 2,
              padding: 1,
              alignItems: "center",
              overflowY: "auto",
            }}
          >
            <SearchBar
              onSearch={handleSearch}
              initialValue=""
              placeholder="Search objects or events..."
            />

            {searchLoading ? (
              <Typography>Searching...</Typography>
            ) : searchError ? (
              <Typography color="danger">Error: {searchError}</Typography>
            ) : error ? (
              <Typography color="danger">Error: {error}</Typography>
            ) : (
              <TrackingSection videos={videos} />
            )}
          </Box>

          <Box
            sx={{
              width: "auto",
              height: "auto",
              padding: 2,
            }}
          >
            <ObjectItem items={items} onClick={handleObjectSelect} />
          </Box>
        </Box>
      </Box>
      <FloatingChatButton />
    </>
  );
}

export default App;
