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

type ObjectProps = {
  name?: string;
  subItems?: string[];
};

type ListItems = {
  items: ObjectProps[];
  isVisible?: boolean; // Prop để điều khiển ẩn/hiện từ bên ngoài (tùy chọn)
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
  isVisible: initialVisible = true, // Mặc định là hiện
}: ListItems) => {
  // State để theo dõi trạng thái của các checkbox
  const [checkedItems, setCheckedItems] = useState<{
    [key: string]: boolean;
  }>({});

  // State để điều khiển ẩn/hiện AccordionGroup
  const [visible, setVisible] = useState(initialVisible);

  // Hàm xử lý khi checkbox thay đổi
  const handleCheckboxChange = (itemName: string, subItem: string) => {
    const key = `${itemName}-${subItem}`;
    setCheckedItems((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  // Hàm toggle ẩn/hiện
  const toggleVisibility = () => {
    setVisible((prev) => !prev);
  };

  return (
    <>
      {/* Nút để toggle ẩn/hiện */}
      <Box sx={{ display: "flex", justifyContent: "center", mb: 1 }}>
        <Button onClick={toggleVisibility}>
          {visible ? "Hide Accordion" : "Show Accordion"}
        </Button>
      </Box>
      {visible && (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            flexDirection: "column",
          }}
        >
          {/* AccordionGroup chỉ hiển thị khi visible là true */}

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
              position: "relative",
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
    </>
  );
};

export default ObjectItem;
