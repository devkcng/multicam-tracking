import { Card, CardContent, CardCover, Typography } from "@mui/joy";
type VideoCardProps = {
  videoSrc: string;
  title: string;
};
const VideoCard = ({ videoSrc, title }: VideoCardProps) => {
  return (
    <>
      <Card component="li" sx={{ minWidth: 300, border: "none" }}>
        <CardCover>
          <video
            style={{
              borderRadius: "8px",
              width: "100%",
              height: "100%",
            }}
            autoPlay
            loop
            muted
          >
            <source src={videoSrc || "/video/demo2.mp4"} type="video/mp4" />
          </video>
        </CardCover>
        <CardContent
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Typography
            level="body-lg"
            textColor="#fff"
            sx={{
              background: "black",
              borderRadius: "8px",
              width: "fit-content",
              paddingX: 1,
              textAlign: "center",
              fontWeight: "lg",
              fontSize: "xs",
              mt: { xs: 12, sm: 18 },
            }}
          >
            {title || "cam1"}
          </Typography>
        </CardContent>
      </Card>
    </>
  );
};

export default VideoCard;
