import { useState, ChangeEvent } from "react";
import {
  Button,
  Modal,
  ModalDialog,
  ModalClose,
  Stack,
  Typography,
  CircularProgress,
  Alert,
} from "@mui/joy";
import apiClient from "../config";
import FileUploadIcon from "@mui/icons-material/FileUpload";

const VideoUploader = () => {
  const [open, setOpen] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("video/")) {
        setError("Please select a video file.");
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError("");
      setMessage("");
      console.log("Selected file:", file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a video file.");
      return;
    }

    setUploading(true);
    setError("");
    setMessage("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    console.log("FormData contents before sending:");
    for (const [key, value] of formData.entries()) {
      console.log(`${key}: ${value}`);
    }

    try {
      const { data } = await apiClient.post("/upload-video", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setMessage(data.message);
      setSelectedFile(null);
      setOpen(false);
    } catch (error: any) {
      setError(
        error.response?.data?.detail?.msg ||
          error.message ||
          "Failed to upload video."
      );
      console.error("Upload error:", error.response || error);
    } finally {
      setUploading(false);
    }
  };

  const truncateFileName = (name: string, maxLength: number = 20): string => {
    if (name.length <= maxLength) return name;
    const extension = name.substring(name.lastIndexOf("."));
    const nameWithoutExt = name.substring(0, name.lastIndexOf("."));
    if (nameWithoutExt.length <= maxLength) return name;
    const leftLength = Math.floor((maxLength - 3) / 2);
    const rightLength = maxLength - 3 - leftLength;
    const leftPart = nameWithoutExt.substring(0, leftLength);
    const rightPart = nameWithoutExt.substring(
      nameWithoutExt.length - rightLength
    );
    return `${leftPart}...${rightPart}${extension}`;
  };

  return (
    <>
      <FileUploadIcon
        onClick={() => setOpen(true)}
        sx={{ color: "primary.500", cursor: "pointer", fontSize: 40 }}
      />
      <Modal open={open} onClose={() => setOpen(false)}>
        <ModalDialog>
          <ModalClose />
          <Stack spacing={2} sx={{ padding: 2, minWidth: 400 }}>
            <Typography level="h4" textAlign="center">
              Upload Video
            </Typography>
            <input
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              disabled={uploading}
              style={{ marginBottom: "16px" }}
            />
            {selectedFile && (
              <Typography level="h4" textAlign="center">
                Selected file: {truncateFileName(selectedFile.name, 20)}
              </Typography>
            )}
            <Button
              variant="solid"
              color="primary"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              startDecorator={uploading ? <CircularProgress size="sm" /> : null}
            >
              {uploading ? "Sending..." : "Send"}
            </Button>
            {message && <Alert color="success">{message}</Alert>}
            {error && <Alert color="danger">{error}</Alert>}
          </Stack>
        </ModalDialog>
      </Modal>
    </>
  );
};

export default VideoUploader;
