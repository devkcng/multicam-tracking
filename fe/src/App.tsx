import { Box, Typography } from "@mui/joy";
import SearchBar from "./components/SearchBar";
import VideoCard from "./components/VideoCard";
import ObjectItem from "./components/ObjectItem";
import TrackingSection from "./components/TrackingSection";

function App() {
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
    </>
  );
}

export default App;
