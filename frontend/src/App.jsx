import React, { useState } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import RealtimeTranslator from './components/RealtimeTranslator';
import VideoUpload from './components/VideoUpload';
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  AppBar,
  Toolbar,
  Container
} from '@mui/material';
import TranslateIcon from '@mui/icons-material/Translate';
import VideoCameraFrontIcon from '@mui/icons-material/VideoCameraFront';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTab(newValue);
  };

  return (
    <>
      <AppBar position="static" color="primary" enableColorOnDark>
        <Toolbar>
          <TranslateIcon sx={{ mr: 2 }} />
          <Typography variant="h5" component="div" sx={{ flexGrow: 1 }}>
            Sign Language Translator
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 2, borderRadius: 3 }}>
          <Tabs
            value={tab}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            variant="fullWidth"
            aria-label="translation mode tabs"
          >
            <Tab icon={<VideoCameraFrontIcon />} label="Realtime Camera" id="tab-0" aria-controls="tabpanel-0" />
            <Tab icon={<CloudUploadIcon />} label="Video Upload" id="tab-1" aria-controls="tabpanel-1" />
          </Tabs>
          <TabPanel value={tab} index={0}>
            <RealtimeTranslator />
          </TabPanel>
          <TabPanel value={tab} index={1}>
            <VideoUpload />
          </TabPanel>
        </Paper>
        <Box sx={{ mt: 4, textAlign: 'center', color: 'text.secondary' }}>
          <Typography variant="body2">
            Built for accessibility. Please allow camera access for real-time translation.
          </Typography>
        </Box>
      </Container>
    </>
  );
}
