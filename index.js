const express = require('express');
const multer = require('multer');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const cors = require('cors');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const upload = multer();
app.use(cors());

(async () => {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models');
  await faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
  await faceapi.nets.faceRecognitionNet.loadFromDisk('./models');
  console.log('Models loaded');
})();

app.post('/embed', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'Image required' });
  }

  const img = await canvas.loadImage(req.file.buffer);

  const detection = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) {
    return res.status(400).json({ error: 'No face detected' });
  }

  res.json({
    embedding: Array.from(detection.descriptor),
  });
});

app.listen(3001, () => {
  console.log('Face API running on http://localhost:3001');
});


app.get('/', async (req, res) => {
  return res.send('Hello World');
});