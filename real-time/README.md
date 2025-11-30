# Supertonic Real-Time Conversational TTS

A real-time voice-to-voice conversational AI system built with Supertonic TTS and Ollama. Have natural conversations with AI through voice input and output.

## Features

- 🚀 **Real-time Streaming**: Streams audio chunks as they're generated for low-latency speech
- 🤖 **Ollama Integration**: Real-time conversations with local LLMs
- 🎤 **Voice Input**: Live voice transcription for hands-free conversations
- 📡 **Multiple Protocols**: Supports both Server-Sent Events (SSE) and WebSocket
- ⚡ **Low Latency**: First audio chunk available within seconds
- 🎭 **Voice Styles**: Support for all voice presets (M1, M2, F1, F2)
- 🔧 **Configurable**: Adjustable denoising steps and speech speed
- 💬 **Conversation History**: Maintains context across multiple messages
- 🎯 **User Priority**: AI automatically stops speaking when user starts talking

## Prerequisites

1. **Node.js** (v18 or higher)
2. **Ollama** installed and running with at least one model pulled
3. **Supertonic assets** (ONNX models and voice styles) in `../assets/`

## Quick Start

```bash
# Install dependencies
npm install

# Start Ollama (if not already running)
ollama serve

# Pull a model
ollama pull llama3.2

# Start the server
npm start
```

The server will start on `http://localhost:3001`.

## Usage

Open `conversation-client.html` in your browser to start a voice conversation with the AI.

## License

MIT

