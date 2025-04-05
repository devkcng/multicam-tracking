import { Box, Typography } from "@mui/joy";
import SearchBar from "./components/SearchBar";
import VideoCard from "./components/VideoCard";

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
          <Box
            sx={{
              display: "flex",
              flexWrap: "wrap",
              gap: 2,
              padding: 2,
              justifyContent: "center",
            }}
          >
            <VideoCard></VideoCard>
            <VideoCard></VideoCard>
            <VideoCard></VideoCard>
            <VideoCard></VideoCard>
            <VideoCard></VideoCard>
          </Box>
        </Box>
      </Box>
    </>
  );
}

export default App;
