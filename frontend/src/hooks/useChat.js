import { useState, useCallback, useRef } from 'react'

const API_URL = '/api/chat/'
let nextId = 0

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isStreaming) return

    const userMessage = { id: nextId++, role: 'user', content: text }
    setMessages(prev => [...prev, userMessage])
    setIsStreaming(true)

    const assistantMessage = {
      id: nextId++,
      role: 'assistant',
      movies: [],
      explanation: '',
      intent: null,
      error: null,
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      const controller = new AbortController()
      abortRef.current = controller

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ error: 'Request failed' }))
        throw new Error(err.error || `HTTP ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let eventType = null
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && eventType) {
            const data = JSON.parse(line.slice(6))
            handleSSEEvent(eventType, data, setMessages, setSessionId)
            eventType = null
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages(prev => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last?.role === 'assistant') {
            updated[updated.length - 1] = { ...last, error: err.message }
          }
          return updated
        })
      }
    } finally {
      setIsStreaming(false)
      abortRef.current = null
    }
  }, [isStreaming, sessionId])

  const reset = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    setMessages([])
    setSessionId(null)
    setIsStreaming(false)
  }, [])

  return { messages, isStreaming, sendMessage, sessionId, reset }
}

function handleSSEEvent(event, data, setMessages, setSessionId) {
  switch (event) {
    case 'movies':
      setSessionId(data.session_id)
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            movies: data.movies,
            intent: data.intent,
          }
        }
        return updated
      })
      break
    case 'session':
      setSessionId(data.session_id)
      break
    case 'token':
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            explanation: last.explanation + data.text,
          }
        }
        return updated
      })
      break
    case 'error':
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            error: data.message,
          }
        }
        return updated
      })
      break
    case 'done':
      break
  }
}
