const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Service URLs - will be environment variables for Railway
const WORDLE_BOT_URL = process.env.WORDLE_BOT_URL || 'http://localhost:5000';

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'Backend healthy' });
});

/**
 * Fetch with timeout wrapper
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = 35000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Wordle bot endpoint
 * POST /api/wordle/play
 * Body: { word?: string, model?: string }
 */
app.post('/api/wordle/play', async (req, res) => {
  try {
    const { word, model = 'entropy_maximization' } = req.body;

    // Build query string
    const params = new URLSearchParams();
    params.append('model', model);
    if (word) {
      params.append('word', word.toLowerCase());
    }

    // Call Flask service
    const response = await fetchWithTimeout(
      `${WORDLE_BOT_URL}/play?${params.toString()}`,
      { method: 'GET' },
      35000 // 35 second timeout
    );

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.json(data);
  } catch (error) {
    console.error('Error calling Wordle bot:', error.message);

    if (error.name === 'AbortError') {
      return res.status(504).json({
        success: false,
        error: 'Wordle bot service took too long to respond'
      });
    }

    if (error.message.includes('ECONNREFUSED')) {
      return res.status(503).json({
        success: false,
        error: 'Wordle bot service is unavailable'
      });
    }

    if (error.message.includes('ENOTFOUND')) {
      return res.status(503).json({
        success: false,
        error: 'Cannot reach Wordle bot service'
      });
    }

    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get available models
 * GET /api/wordle/models
 */
app.get('/api/wordle/models', async (req, res) => {
  try {
    const response = await fetchWithTimeout(
      `${WORDLE_BOT_URL}/models`,
      { method: 'GET' },
      5000 // 5 second timeout
    );

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.json(data);
  } catch (error) {
    console.error('Error fetching models:', error.message);
    res.status(500).json({
      success: false,
      error: 'Could not fetch available models'
    });
  }
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

app.listen(PORT, () => {
  console.log(`Backend server running on port ${PORT}`);
  console.log(`Wordle bot service URL: ${WORDLE_BOT_URL}`);
});