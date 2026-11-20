import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from '../Nemofy Agent/src/App.tsx';
import '../Nemofy Agent/src/index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
