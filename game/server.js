const express = require('express')
const app = express()
const port = process.env.PORT || 3100
const path = require('path')
const WebSocket = require('ws')
const http = require('http')

// Create HTTP server
const server = http.createServer(app)

// Attach WebSocket server
const wss = new WebSocket.Server({ server })

// Serve static files
app.use('/', express.static(path.join(__dirname, 'public')))

// WebSocket relay between the RL agent (Python) and the game client (this server's own
// static page, running in a browser). Each server instance serves exactly one game, so at
// most one of each role is connected at a time.
//
// Message format:
//   agent -> server:   {type: "step", action: [q, w, o, p]} | {type: "reset"}
//   server -> browser: forwarded as-is
//   browser -> server: {type: "state", ...fields}
//   server -> agent:   forwarded as-is
let browserSocket = null
let agentSocket = null

wss.on('connection', (ws) => {
  ws.on('message', (data) => {
    let message
    try {
      message = JSON.parse(data)
    } catch (error) {
      console.error('WebSocket: invalid JSON message:', data)
      return
    }

    if (message.type === 'hello' && message.role === 'browser') {
      console.log('Game browser connected')
      browserSocket = ws
      return
    }

    if (message.type === 'state') {
      if (agentSocket && agentSocket.readyState === WebSocket.OPEN) {
        agentSocket.send(JSON.stringify(message))
      }
      return
    }

    if (message.type === 'step' || message.type === 'reset') {
      agentSocket = ws
      if (!browserSocket || browserSocket.readyState !== WebSocket.OPEN) {
        ws.send(JSON.stringify({ error: 'Game browser not connected' }))
        return
      }
      browserSocket.send(JSON.stringify(message))
      return
    }

    console.warn('WebSocket: unrecognized message', message)
  })

  ws.on('close', () => {
    if (ws === browserSocket) {
      console.log('Game browser disconnected')
      browserSocket = null
    }
    if (ws === agentSocket) {
      console.log('RL agent disconnected')
      agentSocket = null
    }
  })
})

server.listen(port, () => console.log(`PORT: ${port}`))
