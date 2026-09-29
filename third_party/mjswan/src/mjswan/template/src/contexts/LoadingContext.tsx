import React, { createContext, useContext, useState, useCallback, useRef, ReactNode } from 'react';

interface LoadingContextType {
  isLoading: boolean;
  loadingMessage: string | null;
  showLoading: (message?: string | null) => void;
  setLoadingMessage: (message: string | null) => void;
  hideLoading: () => void;
}

const LoadingContext = createContext<LoadingContextType | undefined>(undefined);

export const useLoading = () => {
  const context = useContext(LoadingContext);
  if (!context) {
    throw new Error('useLoading must be used within a LoadingProvider');
  }
  return context;
};

interface LoadingProviderProps {
  children: ReactNode;
}

export const LoadingProvider: React.FC<LoadingProviderProps> = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessageState] = useState<string | null>(null);
  const loadingCountRef = useRef(0);

  const showLoading = useCallback((message?: string | null) => {
    loadingCountRef.current += 1;
    if (message !== undefined) {
      setLoadingMessageState(message);
    }
    if (loadingCountRef.current > 0) {
      setIsLoading(true);
    }
  }, []);

  const setLoadingMessage = useCallback((message: string | null) => {
    setLoadingMessageState(message);
  }, []);

  const hideLoading = useCallback(() => {
    loadingCountRef.current = Math.max(0, loadingCountRef.current - 1);
    if (loadingCountRef.current === 0) {
      setIsLoading(false);
      setLoadingMessageState(null);
    }
  }, []);

  return (
    <LoadingContext.Provider value={{ isLoading, loadingMessage, showLoading, setLoadingMessage, hideLoading }}>
      {children}
    </LoadingContext.Provider>
  );
};
