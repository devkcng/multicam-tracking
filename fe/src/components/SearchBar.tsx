import ClearIcon from "@mui/icons-material/Clear";
import SearchIcon from "@mui/icons-material/Search";
import { Box, IconButton, Input } from "@mui/joy";
import React, { useState } from "react";
import VideoUploader from "./VideoUpLoader";
interface SearchBarProps {
  onSearch: (searchTerm: string) => void;
  placeholder?: string;
  initialValue?: string;
}
const SearchBar = ({
  onSearch,
  placeholder = "Search...",
  initialValue = "",
}: SearchBarProps) => {
  const [searchTerm, setSearchTerm] = useState(initialValue);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(searchTerm.trim());
  };

  const handleClear = () => {
    setSearchTerm("");
    onSearch("");
  };

  return (
    <Box
      component="form"
      onSubmit={handleSearch}
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1,
        width: "50vw",
      }}
    >
      {/* Upload button */}
      <VideoUploader></VideoUploader>
      <Input
        startDecorator={<SearchIcon />}
        endDecorator={
          searchTerm && (
            <IconButton
              variant="plain"
              color="neutral"
              onClick={handleClear}
              aria-label="clear search"
            >
              <ClearIcon />
            </IconButton>
          )
        }
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder={placeholder}
        sx={{
          width: "100%",
          "--Input-focusedThickness": "1px",
          "--Input-focusedHighlight": "primary",
        }}
      />
      <IconButton
        type="submit"
        variant="solid"
        color="primary"
        aria-label="search"
      >
        <SearchIcon />
      </IconButton>

      {/* Video Section */}
    </Box>
  );
};

export default SearchBar;
