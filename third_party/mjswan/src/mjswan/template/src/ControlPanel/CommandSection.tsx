import React from 'react';
import { useDisclosure } from '@mantine/hooks';
import { Box, Collapse, Paper } from '@mantine/core';
import { IconChevronDown, IconChevronUp } from '@tabler/icons-react';

interface CommandSectionProps {
  /** Section label displayed on the border */
  label: string;
  /** Whether the section is expanded by default */
  expandByDefault?: boolean;
  /** Child content to display when expanded */
  children: React.ReactNode;
}

/**
 * CommandSection - A collapsible section with a label on the border
 * Similar to viser's Folder component
 */
export function CommandSection({
  label,
  expandByDefault = true,
  children,
}: CommandSectionProps) {
  const [opened, { toggle }] = useDisclosure(expandByDefault);
  const ToggleIcon = opened ? IconChevronUp : IconChevronDown;

  return (
    <Paper
      withBorder
      style={{
        borderWidth: '1px',
        position: 'relative',
        marginLeft: 'var(--mantine-spacing-xs)',
        marginRight: 'var(--mantine-spacing-xs)',
        marginTop: 'var(--mantine-spacing-xs)',
        marginBottom: 'var(--mantine-spacing-xs)',
        paddingBottom: 'calc(var(--mantine-spacing-xs) - 0.5em)',
      }}
    >
      <Paper
        onClick={toggle}
        style={{
          fontSize: '0.875em',
          position: 'absolute',
          padding: '0 0.375em',
          top: 0,
          left: '0.375em',
          transform: 'translateY(-50%)',
          userSelect: 'none',
          fontWeight: 500,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
        }}
      >
        {label}
        <ToggleIcon
          style={{
            width: '0.9em',
            height: '0.9em',
            strokeWidth: 3,
            top: '0.1em',
            position: 'relative',
            marginLeft: '0.25em',
            marginRight: '-0.1em',
            opacity: 0.5,
          }}
        />
      </Paper>
      <Collapse in={opened}>
        <Box pt="1em">
          {children}
        </Box>
      </Collapse>
      <Collapse in={!opened}>
        <Box p="xs" />
      </Collapse>
    </Paper>
  );
}
