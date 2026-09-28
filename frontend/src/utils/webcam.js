export async function captureFrame(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(blob.arrayBuffer());
    }, 'image/jpeg');
  });
}