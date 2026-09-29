import React from 'react';
import { Loader as MantineLoader } from '@mantine/core';
import { useLoading } from '../contexts/LoadingContext';
import './Loader.css';

export const Loader: React.FC = () => {
  const { isLoading, loadingMessage } = useLoading();

  if (!isLoading) {
    return null;
  }

  return (
    <div className="loader-overlay">
      <div className="loader-content">
        <MantineLoader size={64} type="bars" />
        {loadingMessage && <p className="loader-message">{loadingMessage}</p>}
      </div>
    </div>
  );
};
