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

// WebSocket API for RL agent
// Message format: {action: [q, w, o, p], step: N}
// Response format: {obs: [...], reward: X, terminated: bool, truncated: bool, info: {...}, state: {...}}
wss.on('connection', (ws) => {
  console.log('RL Agent connected via WebSocket')

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data)
      const { action, clientId } = message

      if (!action || !Array.isArray(action) || action.length !== 4) {
        ws.send(JSON.stringify({ error: 'Invalid action format' }))
        return
      }

      // Get current game state from all clients
      // In a real implementation, this would communicate with the game client
      // For now, we'll respond with a placeholder that the client fills in
      const response = {
        type: 'action',
        action: action,
        clientId: clientId || 'default'
      }

      // Broadcast to all connected game clients
      wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN && client !== ws) {
          client.send(JSON.stringify(response))
        }
      })

      // TODO: Receive response from game client with state/reward/done
      // For now, immediately send a dummy response
      // In production, wait for game client to respond with actual data
      ws.send(JSON.stringify({
        obs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        reward: 0,
        terminated: false,
        truncated: false,
        info: {},
        state: {}
      }))
    } catch (error) {
      console.error('WebSocket error:', error)
      ws.send(JSON.stringify({ error: 'Internal server error' }))
    }
  })

  ws.on('close', () => {
    console.log('RL Agent disconnected')
  })
})

server.listen(port, () => console.log(`PORT: ${port}`))
