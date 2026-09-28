import React, { useRef, useEffect, useState } from 'react';
import { 
  Typography, 
  Button, 
  Box, 
  Alert, 
  CircularProgress,
  Paper,
  Chip,
  LinearProgress
} from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import TranslateIcon from '@mui/icons-material/Translate';

const RealtimeTranslator = () => {
  const videoRef = useRef();
  const canvasRef = useRef();
  const wsRef = useRef();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [stream, setStream] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [translationResult, setTranslationResult] = useState(null);
  const [sequenceProgress, setSequenceProgress] = useState(0);

  const startCamera = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
        setIsStreamActive(true);
        
        // Start WebSocket connection
        startWebSocket();
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError(err.message || 'Failed to access camera. Please check permissions.');
    } finally {
      setIsLoading(false);
    }
  };

  const startWebSocket = () => {
    console.log('Starting WebSocket connection...');
    const ws = new WebSocket('ws://localhost:8000/api/realtime/ws');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected successfully');
      setIsConnected(true);
      startFrameCapture();
    };

    ws.onmessage = (event) => {
      console.log('Received WebSocket message:', event.data);
      try {
        const data = JSON.parse(event.data);
        
        if (data.status === 'building_sequence') {
          setSequenceProgress((data.frames_collected / data.frames_needed) * 100);
          setTranslationResult(null);
        } else if (data.status === 'error') {
          console.error('Backend error:', data.error);
          setError(`Backend error: ${data.error}`);
        } else {
          setTranslationResult(data);
          setSequenceProgress(0);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Failed to connect to translation service');
    };
  };

  const startFrameCapture = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');

    const captureFrame = () => {
      if (video && isConnected && wsRef.current?.readyState === WebSocket.OPEN) {
        // Check if video has valid dimensions
        if (video.videoWidth === 0 || video.videoHeight === 0) {
          console.log('Video not ready yet, retrying...');
          setTimeout(captureFrame, 100);
          return;
        }

        // Draw video frame to canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        console.log(`Capturing frame: ${canvas.width}x${canvas.height}`);

        // Convert canvas to blob and send
        canvas.toBlob((blob) => {
          if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
            console.log(`Sending blob: ${blob.size} bytes`);
            blob.arrayBuffer().then(buffer => {
              console.log(`Sending buffer: ${buffer.byteLength} bytes`);
              wsRef.current.send(buffer);
            }).catch(error => {
              console.error('Error converting blob to buffer:', error);
            });
          } else {
            console.log('Blob is null or WebSocket not open');
          }
        }, 'image/jpeg', 0.8);
      } else {
        console.log('Video or WebSocket not ready:', {
          video: !!video,
          isConnected,
          wsReadyState: wsRef.current?.readyState
        });
      }
      
      // Continue capturing at 30fps
      if (isStreamActive && isConnected) {
        setTimeout(captureFrame, 33); // ~30fps
      }
    };

    captureFrame();
  };

  const sendTestFrame = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('Sending test frame...');
      // Create a simple test image
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      canvas.width = 100;
      canvas.height = 100;
      ctx.fillStyle = 'red';
      ctx.fillRect(0, 0, 100, 100);
      
      canvas.toBlob((blob) => {
        if (blob) {
          console.log(`Sending test blob: ${blob.size} bytes`);
          blob.arrayBuffer().then(buffer => {
            console.log(`Sending test buffer: ${buffer.byteLength} bytes`);
            wsRef.current.send(buffer);
          });
        }
      }, 'image/jpeg', 0.8);
    } else {
      console.log('WebSocket not open for test frame');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setIsStreamActive(false);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setTranslationResult(null);
    setSequenceProgress(0);
  };

  useEffect(() => {
    startCamera();
    
    // Cleanup function
    return () => {
      stopCamera();
    };
  }, []);

  const handleToggleCamera = () => {
    if (isStreamActive) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Live Translation from Camera
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper 
        elevation={3} 
        sx={{ 
          p: 2, 
          mb: 2, 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          minHeight: '300px',
          backgroundColor: '#f5f5f5'
        }}
      >
        {isLoading ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <CircularProgress />
            <Typography>Initializing camera...</Typography>
          </Box>
        ) : (
          <Box sx={{ position: 'relative', width: '100%', maxWidth: '640px' }}>
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline
              muted
              style={{ 
                width: '100%',
                borderRadius: '8px',
                backgroundColor: '#000'
              }} 
            />
            <canvas 
              ref={canvasRef} 
              style={{ display: 'none' }}
            />
            
            {/* Connection status overlay */}
            <Box sx={{ 
              position: 'absolute', 
              top: 8, 
              right: 8,
              display: 'flex',
              gap: 1
            }}>
              <Chip 
                label={isConnected ? "Connected" : "Disconnected"}
                color={isConnected ? "success" : "error"}
                size="small"
              />
              {isStreamActive && (
                <Chip 
                  label="Recording"
                  color="primary"
                  size="small"
                />
              )}
            </Box>
          </Box>
        )}
      </Paper>

      {/* Translation Progress */}
      {sequenceProgress > 0 && sequenceProgress < 100 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" gutterBottom>
            Building translation sequence... ({Math.round(sequenceProgress)}%)
          </Typography>
          <LinearProgress variant="determinate" value={sequenceProgress} />
        </Box>
      )}

      {/* Translation Results */}
      {translationResult && (
        <Paper elevation={2} sx={{ p: 2, mb: 2, backgroundColor: '#f0f8ff' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <TranslateIcon color="primary" />
            <Typography variant="h6" color="primary">
              Translation Result
            </Typography>
          </Box>
          
          <Typography variant="body1" sx={{ mb: 1, fontWeight: 'bold' }}>
            {translationResult.english_translation}
          </Typography>
          
          {translationResult.gloss_sequence.length > 0 && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                Detected signs:
              </Typography>
              {translationResult.gloss_sequence.map((gloss, index) => (
                <Chip 
                  key={index}
                  label={gloss}
                  size="small"
                  variant="outlined"
                />
              ))}
            </Box>
          )}
          
          <Typography variant="caption" color="text.secondary">
            Confidence: {Math.round(translationResult.confidence * 100)}%
          </Typography>
        </Paper>
      )}

      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button
          variant="contained"
          color={isStreamActive ? "error" : "primary"}
          onClick={handleToggleCamera}
          startIcon={isStreamActive ? <VideocamOffIcon /> : <VideocamIcon />}
          disabled={isLoading}
        >
          {isStreamActive ? 'Stop Camera' : 'Start Camera'}
        </Button>
        
        <Button
          variant="outlined"
          onClick={sendTestFrame}
          disabled={!isConnected}
        >
          Send Test Frame
        </Button>
      </Box>

      {isStreamActive && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Camera is active. Translation will appear once the sequence is built (1 second).
        </Alert>
      )}
    </Box>
  );
};

export default RealtimeTranslator;
