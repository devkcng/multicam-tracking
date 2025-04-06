import { Radio, Box, Button } from "@mui/joy";
import React, { useState } from "react";
import FilterListAltIcon from "@mui/icons-material/FilterListAlt";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";

export type ObjectProps = {
  id?: string;
};

type ListItems = {
  items: ObjectProps[];
  isVisible?: boolean;
  onClick: (selectedItem: string) => void;
};

const ObjectItem = ({
  items = [{ id: "1" }, { id: "2" }, { id: "3" }, { id: "4" }],
  isVisible: initialVisible = true,
  onClick,
}: ListItems) => {
  const [selectedItem, setSelectedItem] = useState<string>("");
  const [visible, setVisible] = useState(initialVisible);

  const handleRadioChange = (itemName: string) => {
    setSelectedItem(itemName);
  };

  const handleSave = () => {
    onClick(selectedItem);
  };

  const toggleVisibility = () => {
    setVisible((prev) => !prev);
  };

  return (
    <Box
      sx={{
        position: "relative",
        alignItems: "center",
        display: "flex",
        flexDirection: "column",
        gap: 1,
      }}
    >
      <Button
        onClick={toggleVisibility}
        sx={{
          position: "absolute",
          right: visible ? "auto" : "-15px",
          top: visible ? "0" : "50%",
          transform: visible ? "none" : "translateY(-50%)",
          transition: "all 0.3s ease",
          minWidth: visible ? "auto" : "40px",
          width: visible ? "auto" : "40px",
          height: visible ? "auto" : "40px",
          borderRadius: visible ? "4px" : "50%",
          padding: visible ? "6px 16px" : "0",
          zIndex: 1,
          ...(visible
            ? { margin: "0 auto", display: "block" }
            : {
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }),
        }}
      >
        {visible ? <VisibilityOffIcon /> : <FilterListAltIcon />}
      </Button>

      {visible && (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            flexDirection: "column",
            paddingTop: "40px",
          }}
        >
          <Box
            sx={{
              backgroundColor: "#EEEEEE",
              maxWidth: 400,
              maxHeight: "400px",
              padding: 2,
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 1,
            }}
          >
            {items.map((item, index) => (
              <Box
                key={index}
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  paddingBottom: 2,
                }}
              >
                <Radio
                  checked={selectedItem === item.id}
                  onChange={() => handleRadioChange(item.id || "")}
                />
                id:{item.id}
              </Box>
            ))}
          </Box>

          <Button onClick={handleSave}>Save</Button>
        </Box>
      )}
    </Box>
  );
};

export default ObjectItem;
