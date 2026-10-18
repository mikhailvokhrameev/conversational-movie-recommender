import { useState, useCallback, useRef } from 'react'

const API_URL = '/api/chat/'
let nextId = 0

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const abortRef = useRef(null)
  const sessionTokenRef = useRef(null)
  // The card in view in the carousel the user last looked at; the server
  // ignores it unless it belongs to the latest shown list.
  const selectedRef = useRef({ messageId: null, tmdbId: null })
  // Message id of the latest shown list; only its carousel may set the selection.
  const latestListRef = useRef(null)

  const selectMovie = useCallback((messageId, movie) => {
    if (messageId !== latestListRef.current) return
    selectedRef.current = { messageId, tmdbId: movie?.tmdb_id ?? null }
  }, [])

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
      descriptions: [],
      intent: null,
      error: null,
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      const controller = new AbortController()
      abortRef.current = controller

      const headers = { 'Content-Type': 'application/json' }
      if (sessionTokenRef.current) headers['X-Session-Token'] = sessionTokenRef.current

      const response = await fetch(API_URL, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
          selected_tmdb_id: selectedRef.current.tmdbId,
        }),
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
            if (eventType === 'movies' && data.movies?.length > 0) {
              latestListRef.current = assistantMessage.id
              selectedRef.current = { messageId: assistantMessage.id, tmdbId: null }
            }
            handleSSEEvent(eventType, data, setMessages, setSessionId, sessionTokenRef)
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
    sessionTokenRef.current = null
    selectedRef.current = { messageId: null, tmdbId: null }
    latestListRef.current = null
    setIsStreaming(false)
  }, [])

  return { messages, isStreaming, sendMessage, selectMovie, sessionId, reset }
}

function handleSSEEvent(event, data, setMessages, setSessionId, sessionTokenRef) {
  switch (event) {
    case 'movies':
      setSessionId(data.session_id)
      sessionTokenRef.current = data.session_token
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
      sessionTokenRef.current = data.session_token
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
    case 'intro_final':
      // The reply ignored the [[n]] markers: keep only its opening line, the
      // cards fall back to catalog descriptions.
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            explanation: data.text,
          }
        }
        return updated
      })
      break
    case 'blurb':
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last?.role === 'assistant') {
          const descriptions = [...(last.descriptions || [])]
          while (descriptions.length <= data.index) descriptions.push('')
          descriptions[data.index] = descriptions[data.index] + data.text
          updated[updated.length - 1] = {
            ...last,
            descriptions,
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
