import { Box, Flex, Text } from "@mantine/core";
import React from "react";

interface LabeledInputProps {
  id?: string;
  label: string;
  children: React.ReactNode;
}

/** GUI input with a label horizontally placed to the left of it. */
export function LabeledInput({ id, label, children }: LabeledInputProps) {
  return (
    <Box pb="0.5em" px="xs">
      <Flex align="center">
        <Box
          pr="xs"
          style={{
            width: "7.25em",
            flexShrink: 0,
            position: "relative",
          }}
        >
          <Text
            c="dimmed"
            style={{
              fontSize: "0.875em",
              fontWeight: 450,
              lineHeight: "1.375em",
              letterSpacing: "-0.75px",
              width: "100%",
              boxSizing: "content-box",
            }}
          >
            <label htmlFor={id}>{label}</label>
          </Text>
        </Box>
        <Box style={{ flexGrow: 1 }}>{children}</Box>
      </Flex>
    </Box>
  );
}
