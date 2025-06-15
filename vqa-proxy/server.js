const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fetch = require('node-fetch');
const FormData = require('form-data');
const app = express();
const upload = multer();

app.use(cors());

app.post('/proxy', upload.single('image'), async (req, res) => {
  try {
    const form = new FormData();
    form.append('image', req.file.buffer, req.file.originalname);
    form.append('question', req.body.question);
  
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: form,
      headers: form.getHeaders(),
    });

    const data = await response.text();
    res.set('Content-Type', 'application/json');
    res.send(data);
  } catch (err) {
    res.status(500).json({ error: err.toString() });
  }
});

app.listen(3000, () => console.log('Proxy running on port 3000'));