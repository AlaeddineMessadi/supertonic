import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

import {
  loadTextToSpeech,
  loadVoiceStyle,
  chunkText,
} from '../nodejs/helper.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// ============================================================================
// Logging Utility
// ============================================================================
const LogLevel = {
  ERROR: 0,
  WARN: 1,
  INFO: 2,
  DEBUG: 3,
};

const LOG_LEVEL = process.env.LOG_LEVEL
  ? LogLevel[process.env.LOG_LEVEL.toUpperCase()] ?? LogLevel.INFO
  : LogLevel.INFO;

function formatTimestamp() {
  return new Date().toISOString();
}

function log(level, message, ...args) {
  const levelNames = ['ERROR', 'WARN', 'INFO', 'DEBUG'];
  if (level <= LOG_LEVEL) {
    const prefix = `[${formatTimestamp()}] [${levelNames[level]}]`;
    const logMethod = level === LogLevel.ERROR ? console.error : console.log;
    logMethod(prefix, message, ...args);
  }
}

const logger = {
  error: (message, ...args) => log(LogLevel.ERROR, message, ...args),
  warn: (message, ...args) => log(LogLevel.WARN, message, ...args),
  info: (message, ...args) => log(LogLevel.INFO, message, ...args),
  debug: (message, ...args) => log(LogLevel.DEBUG, message, ...args),
};

// ============================================================================
// Error Handlers
// ============================================================================
process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', error);
  logger.error('Stack:', error.stack);
  setTimeout(() => {
    process.exit(1);
  }, 1000);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise);
  logger.error('Reason:', reason);
  if (reason instanceof Error) {
    logger.error('Stack:', reason.stack);
  }
});

// ============================================================================
// Middleware
// ============================================================================
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Request logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const originalSend = res.send;

  res.send = function (body) {
    const duration = Date.now() - start;
    logger.info(
      `${req.method} ${req.path} - ${res.statusCode} - ${duration}ms - ${req.ip}`
    );
    return originalSend.call(this, body);
  };

  next();
});

// Configuration
const ONNX_DIR = path.resolve(__dirname, '../assets/onnx');
const VOICE_STYLES_DIR = path.resolve(__dirname, '../assets/voice_styles');
const DEFAULT_VOICE = 'M1.json';
const DEFAULT_STEPS = 3;
const DEFAULT_SPEED = 1.4;

// Global TTS instance (loaded once at startup)
let textToSpeech = null;
let defaultStyle = null;

// Initialize TTS
async function initializeTTS() {
  try {
    logger.info('Loading TTS models...');
    const startTime = Date.now();

    textToSpeech = await loadTextToSpeech(ONNX_DIR, false);

    const defaultStylePath = path.join(VOICE_STYLES_DIR, DEFAULT_VOICE);
    defaultStyle = loadVoiceStyle([defaultStylePath], false);

    const loadTime = Date.now() - startTime;
    logger.info(`TTS models loaded successfully! (${loadTime}ms)`);
    return true;
  } catch (error) {
    logger.error('Failed to initialize TTS:', error);
    logger.error('Stack:', error.stack);
    return false;
  }
}

/**
 * Convert audio samples to WAV buffer
 */
function audioToWavBuffer(audioData, sampleRate) {
  const bitsPerSample = 16;
  const numChannels = 1;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const dataSize = (audioData.length * bitsPerSample) / 8;

  const buffer = Buffer.alloc(44 + dataSize);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);

  // fmt chunk
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);

  // data chunk
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);

  // Write audio data
  for (let i = 0; i < audioData.length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    const intSample = Math.floor(sample * 32767);
    buffer.writeInt16LE(intSample, 44 + i * 2);
  }

  return buffer;
}

/**
 * Stream audio chunks as they're generated
 */
async function streamAudioChunks(text, style, totalStep, speed, res) {
  const chunks = chunkText(text, 0);
  let totalDuration = 0;
  let chunkIndex = 0;

  // Send initial metadata
  res.write(
    `data: ${JSON.stringify({ type: 'start', totalChunks: chunks.length })}\n\n`
  );

  for (let i = 0; i < chunks.length; i++) {
    try {
      // Generate audio for this chunk
      const result = await textToSpeech._infer(
        [chunks[i]],
        style,
        totalStep,
        speed
      );
      const duration = result.duration[0];
      const wavLen = Math.floor(textToSpeech.sampleRate * duration);
      const wavChunk = result.wav.slice(0, wavLen);

      // Convert to WAV buffer
      const wavBuffer = audioToWavBuffer(wavChunk, textToSpeech.sampleRate);

      // Send chunk metadata
      res.write(
        `data: ${JSON.stringify({
          type: 'chunk',
          chunkIndex: i + 1,
          totalChunks: chunks.length,
          duration: duration,
          sampleRate: textToSpeech.sampleRate,
        })}\n\n`
      );

      // Send audio data as base64
      const audioBase64 = wavBuffer.toString('base64');
      res.write(
        `data: ${JSON.stringify({
          type: 'audio',
          chunkIndex: i + 1,
          data: audioBase64,
        })}\n\n`
      );

      totalDuration += duration;
      chunkIndex++;

      // Add silence between chunks (except for last chunk)
      if (i < chunks.length - 1) {
        const silenceDuration = 0.3;
        const silenceLen = Math.floor(
          silenceDuration * textToSpeech.sampleRate
        );
        const silence = new Array(silenceLen).fill(0);
        const silenceBuffer = audioToWavBuffer(
          silence,
          textToSpeech.sampleRate
        );
        const silenceBase64 = silenceBuffer.toString('base64');

        res.write(
          `data: ${JSON.stringify({
            type: 'silence',
            duration: silenceDuration,
            data: silenceBase64,
          })}\n\n`
        );

        totalDuration += silenceDuration;
      }
    } catch (error) {
      res.write(
        `data: ${JSON.stringify({
          type: 'error',
          message: error.message,
          chunkIndex: i + 1,
        })}\n\n`
      );
      break;
    }
  }

  // Send completion
  res.write(
    `data: ${JSON.stringify({
      type: 'end',
      totalDuration: totalDuration,
      totalChunks: chunkIndex,
    })}\n\n`
  );

  res.end();
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    ttsLoaded: textToSpeech !== null,
    timestamp: new Date().toISOString(),
  });
});

// Get available voice styles
app.get('/voices', (req, res) => {
  try {
    const files = fs
      .readdirSync(VOICE_STYLES_DIR)
      .filter((f) => f.endsWith('.json'))
      .map((f) => f.replace('.json', ''));
    res.json({ voices: files });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// SSE Streaming endpoint
app.post('/stream', async (req, res) => {
  if (!textToSpeech) {
    return res.status(503).json({ error: 'TTS not initialized' });
  }

  const {
    text,
    voice = DEFAULT_VOICE,
    steps = DEFAULT_STEPS,
    speed = DEFAULT_SPEED,
  } = req.body;

  if (!text || typeof text !== 'string') {
    return res.status(400).json({ error: 'Text is required' });
  }

  // Load voice style
  let style;
  try {
    const voicePath = path.join(
      VOICE_STYLES_DIR,
      voice.endsWith('.json') ? voice : `${voice}.json`
    );
    if (!fs.existsSync(voicePath)) {
      return res.status(400).json({ error: `Voice style not found: ${voice}` });
    }
    style = loadVoiceStyle([voicePath], false);
  } catch (error) {
    return res
      .status(500)
      .json({ error: `Failed to load voice style: ${error.message}` });
  }

  // Set up SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');

  // Stream audio chunks
  await streamAudioChunks(text, style, steps, speed, res);
});

// ============================================================================
// Error Handling Middleware (must be last)
// ============================================================================
app.use((err, req, res, next) => {
  logger.error('Express error handler:', err);
  logger.debug('Request that caused error:', {
    method: req.method,
    path: req.path,
    body: req.body ? JSON.stringify(req.body).substring(0, 200) : 'none',
  });

  const isDevelopment = process.env.NODE_ENV !== 'production';
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error',
    ...(isDevelopment && { stack: err.stack }),
  });
});

// 404 handler
app.use((req, res) => {
  logger.warn(`404 - Route not found: ${req.method} ${req.path}`);
  res.status(404).json({ error: 'Route not found' });
});

// ============================================================================
// Start server
// ============================================================================
const PORT = process.env.PORT || 3001;

async function startServer() {
  const initialized = await initializeTTS();
  if (!initialized) {
    logger.error('Failed to initialize TTS. Server will not start.');
    process.exit(1);
  }

  app.listen(PORT, () => {
    logger.info(`\n🚀 Supertonic Real-Time TTS Server`);
    logger.info(`   Listening on http://localhost:${PORT}`);
    logger.info(`   SSE Endpoint: POST http://localhost:${PORT}/stream`);
    logger.info(`   Health: http://localhost:${PORT}/health\n`);
  });
}

startServer().catch((error) => {
  logger.error('Failed to start server:', error);
  process.exit(1);
});

