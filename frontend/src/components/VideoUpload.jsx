import React, { useState } from 'react';
import { 
  Typography, 
  Button, 
  Box, 
  Alert, 
  Paper,
  LinearProgress
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

const VideoUpload = () => {
  const [videoURL, setVideoURL] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setIsUploading(true);
      setUploadProgress(0);
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 100);

      try {
        // Simulate file processing
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setVideoURL(URL.createObjectURL(file));
        setUploadProgress(100);
        
        // TODO: send file to backend for translation
        console.log('File uploaded:', file.name);
        
      } catch (error) {
        console.error('Upload error:', error);
      } finally {
        setIsUploading(false);
        setTimeout(() => setUploadProgress(0), 2000);
      }
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Upload Video for Translation
      </Typography>
      
      <Paper 
        elevation={3} 
        sx={{ 
          p: 3, 
          mb: 2, 
          display: 'flex', 
          flexDirection: 'column',
          alignItems: 'center',
          gap: 2,
          minHeight: '200px',
          backgroundColor: '#f5f5f5'
        }}
      >
        {!videoURL ? (
          <>
            <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
            <Typography variant="body1" color="text.secondary" textAlign="center">
              Select a video file to upload and translate
            </Typography>
            <Button
              variant="contained"
              component="label"
              startIcon={<CloudUploadIcon />}
              disabled={isUploading}
            >
              Choose Video File
              <input 
                hidden 
                accept="video/*" 
                type="file" 
                onChange={handleFileChange}
                disabled={isUploading}
              />
            </Button>
          </>
        ) : (
          <>
            <video 
              controls 
              src={videoURL} 
              style={{ 
                width: '100%', 
                maxWidth: '640px',
                borderRadius: '8px',
                backgroundColor: '#000'
              }} 
            />
            <Button
              variant="outlined"
              component="label"
              startIcon={<CloudUploadIcon />}
              onClick={() => setVideoURL(null)}
            >
              Upload Another Video
              <input 
                hidden 
                accept="video/*" 
                type="file" 
                onChange={handleFileChange}
              />
            </Button>
          </>
        )}
      </Paper>

      {isUploading && (
        <Box sx={{ width: '100%', mb: 2 }}>
          <Typography variant="body2" gutterBottom>
            Uploading video...
          </Typography>
          <LinearProgress variant="determinate" value={uploadProgress} />
        </Box>
      )}

      {videoURL && !isUploading && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Video uploaded successfully. Translation features will be available once the backend is connected.
        </Alert>
      )}
    </Box>
  );
};

export default VideoUpload;
