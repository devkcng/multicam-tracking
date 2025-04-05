import {
  Accordion,
  AccordionDetails,
  AccordionGroup,
  AccordionSummary,
  Checkbox,
  Box,
  Button,
} from "@mui/joy";
import React, { useState } from "react";
import FilterListAltIcon from "@mui/icons-material/FilterListAlt";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
type ObjectProps = {
  name?: string;
  subItems?: string[];
};

type ListItems = {
  items: ObjectProps[];
  isVisible?: boolean;
};

const ObjectItem = ({
  items = [
    {
      name: "id:1",
      subItems: ["cam1", "cam2", "cam3"],
    },
    {
      name: "id:2",
      subItems: ["cam1", "cam2", "cam3"],
    },
    {
      name: "id:3",
      subItems: ["cam1", "cam2", "cam3"],
    },
    {
      name: "id:4",
      subItems: ["cam1", "cam2", "cam3"],
    },
  ],
  isVisible: initialVisible = true,
}: ListItems) => {
  const [checkedItems, setCheckedItems] = useState<{
    [key: string]: boolean;
  }>({});
  const [visible, setVisible] = useState(initialVisible);

  const handleCheckboxChange = (itemName: string, subItem: string) => {
    const key = `${itemName}-${subItem}`;
    setCheckedItems((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
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
      {/* Nút toggle với style động */}
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
        {visible ? (
          <VisibilityOffIcon></VisibilityOffIcon>
        ) : (
          <FilterListAltIcon></FilterListAltIcon>
        )}
      </Button>

      {visible && (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            flexDirection: "column",
            paddingTop: "40px", // Để tránh đè lên button khi visible
          }}
        >
          <AccordionGroup
            sx={{
              maxWidth: 400,
              height: "auto",
              maxHeight: "400px",
              alignItems: "center",
              display: "flex",
              gap: 1,
              padding: 2,
              overflowY: "auto",
            }}
          >
            {items.map((item, index) => (
              <Accordion key={index}>
                <AccordionSummary>{item.name}</AccordionSummary>
                {item.subItems?.map((subItem, subIndex) => {
                  const checkboxKey = `${item.name}-${subItem}`;
                  return (
                    <AccordionDetails key={subIndex}>
                      <Box
                        sx={{ display: "flex", alignItems: "center", gap: 1 }}
                      >
                        <Checkbox
                          checked={checkedItems[checkboxKey] || false}
                          onChange={() =>
                            handleCheckboxChange(
                              item.name || `id:${index}`,
                              subItem
                            )
                          }
                        />
                        {subItem}
                      </Box>
                    </AccordionDetails>
                  );
                })}
              </Accordion>
            ))}
          </AccordionGroup>

          <Button>Save</Button>
        </Box>
      )}
    </Box>
  );
};

export default ObjectItem;
