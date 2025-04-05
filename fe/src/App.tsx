import { Box, Typography } from "@mui/joy";
import { useEffect } from "react";
import FloatingChatButton from "./components/FloatingChatButton";
import ObjectItem from "./components/ObjectItem";
import SearchBar from "./components/SearchBar";
import TrackingSection from "./components/TrackingSection";
import apiClient from "./config";

function App() {
  useEffect(() => {
    const callAPI = async () => {
      try {
        const response = await apiClient.get("/");
        console.log("API data:", response.data.message);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };
    callAPI();
  }, []);
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
        {/* Main section */}
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
            <SearchBar onSearch={() => {}} initialValue=""></SearchBar>

            <TrackingSection></TrackingSection>
          </Box>
          {/* Filter section */}
          <Box
            sx={{
              width: "auto",
              height: "auto",
              padding: 2,
            }}
          >
            <ObjectItem></ObjectItem>
          </Box>
        </Box>
      </Box>
      <FloatingChatButton />
    </>
  );
}

export default App;
