/**
 * Ollama adapter for Claude Code.
 *
 * Converts between the internal Anthropic-style message format and
 * Ollama's native /api/chat endpoint. Uses the native API (not the
 * OpenAI-compatible layer) to get full control over num_ctx and num_predict.
 * Handles streaming, tool calling, and message normalization.
 *
 * Environment variables:
 *   OLLAMA_NUM_CTX       — Context window size in tokens (default: 131072 = 128K)
 *   OLLAMA_MAX_TOKENS    — Max output tokens per response (default: 16384)
 */
import { randomUUID } from 'crypto'
import type {
  BetaContentBlock,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type {
  AssistantMessage,
  Message,
  MessageContent,
  StreamEvent,
  SystemAPIErrorMessage,
} from '../../types/message.js'
import type { Tool, Tools } from '../../Tool.js'
import {
  normalizeContentFromAPI,
  normalizeMessagesForAPI,
} from '../../utils/messages.js'
import type { SystemPrompt } from '../../utils/systemPromptType.js'
import { getOllamaBaseUrl, getOllamaModel } from '../../utils/model/providers.js'
import { logForDebugging } from '../../utils/debug.js'
import { zodToJsonSchema } from '../../utils/zodToJsonSchema.js'
import type { Options } from './claude.js'

// ---------------------------------------------------------------------------
// Ollama context/output configuration
// ---------------------------------------------------------------------------

const DEFAULT_NUM_CTX = 131_072   // 128K — safe for Gemma 4 26B on 128GB systems
const DEFAULT_MAX_TOKENS = 16_384 // 16K output per turn

function getOllamaNumCtx(): number {
  const val = process.env.OLLAMA_NUM_CTX
  if (val) {
    const n = parseInt(val, 10)
    if (!isNaN(n) && n > 0) return n
  }
  return DEFAULT_NUM_CTX
}

function getOllamaMaxTokens(): number {
  const val = process.env.OLLAMA_MAX_TOKENS
  if (val) {
    const n = parseInt(val, 10)
    if (!isNaN(n) && n > 0) return n
  }
  return DEFAULT_MAX_TOKENS
}

// ---------------------------------------------------------------------------
// Message types for Ollama native /api/chat
// ---------------------------------------------------------------------------

interface OllamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  tool_calls?: OllamaToolCall[]
}

interface OllamaToolCall {
  function: { name: string; arguments: Record<string, unknown> }
}

interface OllamaTool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

/** A single NDJSON chunk from Ollama's /api/chat streaming response. */
interface OllamaChatChunk {
  model: string
  created_at: string
  message: {
    role: string
    content: string
    tool_calls?: {
      function: { name: string; arguments: Record<string, unknown> }
    }[]
  }
  done: boolean
  done_reason?: string
  total_duration?: number
  load_duration?: number
  prompt_eval_count?: number
  prompt_eval_duration?: number
  eval_count?: number
  eval_duration?: number
}

// ---------------------------------------------------------------------------
// Message conversion: Internal (Anthropic-style) → Ollama native chat format
// ---------------------------------------------------------------------------

function systemPromptToString(systemPrompt: SystemPrompt): string {
  if (typeof systemPrompt === 'string') return systemPrompt
  if (Array.isArray(systemPrompt)) {
    return systemPrompt
      .map(block => {
        if (typeof block === 'string') return block
        if ('text' in block) return block.text
        return ''
      })
      .join('\n')
  }
  return String(systemPrompt ?? '')
}

function contentToString(content: MessageContent | undefined): string {
  if (!content) return ''
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map(block => {
        if (typeof block === 'string') return block
        if ('text' in block && block.type === 'text') return (block as { text: string }).text
        return ''
      })
      .filter(Boolean)
      .join('\n')
  }
  return ''
}

/**
 * Extract tool_result blocks from a user message's content array and
 * return them as separate Ollama "tool" role messages, plus the remaining text.
 */
function extractToolResults(msg: Message): OllamaMessage[] {
  const results: OllamaMessage[] = []
  const textParts: string[] = []

  if (!msg.message?.content || typeof msg.message.content === 'string') {
    return [{ role: 'user', content: contentToString(msg.message?.content) }]
  }

  for (const block of msg.message.content as Array<Record<string, unknown>>) {
    if (block.type === 'tool_result') {
      let resultContent = ''
      if (typeof block.content === 'string') {
        resultContent = block.content
      } else if (Array.isArray(block.content)) {
        resultContent = (block.content as Array<Record<string, unknown>>)
          .map(b => ('text' in b ? String(b.text) : ''))
          .join('\n')
      }
      results.push({
        role: 'tool',
        content: resultContent,
      })
    } else if (block.type === 'text' && block.text) {
      textParts.push(String(block.text))
    }
  }

  const out: OllamaMessage[] = []
  if (results.length > 0) {
    out.push(...results)
  }
  if (textParts.length > 0) {
    out.push({ role: 'user', content: textParts.join('\n') })
  }
  if (out.length === 0) {
    out.push({ role: 'user', content: '' })
  }
  return out
}

/**
 * Convert an assistant message with potential tool_use blocks to Ollama native format.
 * Ollama expects tool_calls with function.arguments as an object (not a JSON string).
 */
function convertAssistantMessage(msg: Message): OllamaMessage {
  const out: OllamaMessage = { role: 'assistant', content: '' }
  const textParts: string[] = []
  const toolCalls: OllamaToolCall[] = []

  if (!msg.message?.content || typeof msg.message.content === 'string') {
    out.content = contentToString(msg.message?.content)
    return out
  }

  for (const block of msg.message.content as Array<Record<string, unknown>>) {
    if (block.type === 'text' && block.text) {
      textParts.push(String(block.text))
    } else if (block.type === 'tool_use') {
      let args: Record<string, unknown>
      if (typeof block.input === 'string') {
        try { args = JSON.parse(block.input) } catch { args = {} }
      } else {
        args = (block.input as Record<string, unknown>) ?? {}
      }
      toolCalls.push({
        function: {
          name: String(block.name || ''),
          arguments: args,
        },
      })
    }
  }

  if (textParts.length > 0) out.content = textParts.join('\n')
  if (toolCalls.length > 0) out.tool_calls = toolCalls
  return out
}

export function convertMessagesForOllama(
  messages: Message[],
  systemPrompt: SystemPrompt,
  tools: Tools,
): OllamaMessage[] {
  const ollamaMessages: OllamaMessage[] = []

  const sysText = systemPromptToString(systemPrompt)
  if (sysText) {
    ollamaMessages.push({ role: 'system', content: sysText })
  }

  const normalized = normalizeMessagesForAPI(messages, tools)
  for (const msg of normalized) {
    if (msg.type === 'user' || msg.message?.role === 'user') {
      ollamaMessages.push(...extractToolResults(msg))
    } else if (msg.type === 'assistant' || msg.message?.role === 'assistant') {
      ollamaMessages.push(convertAssistantMessage(msg))
    }
  }

  return ollamaMessages
}

// ---------------------------------------------------------------------------
// Tool conversion: Internal (Anthropic-style) → Ollama native function format
// ---------------------------------------------------------------------------

function getToolParameters(tool: Tool): Record<string, unknown> {
  if ('inputJSONSchema' in tool && tool.inputJSONSchema) {
    return tool.inputJSONSchema as Record<string, unknown>
  }
  if (tool.inputSchema) {
    try {
      return zodToJsonSchema(tool.inputSchema) as Record<string, unknown>
    } catch (e) {
      logForDebugging(`[Ollama] zodToJsonSchema failed for ${tool.name}: ${e}`)
    }
  }
  return { type: 'object', properties: {} }
}

/**
 * Build tool descriptions by calling each tool's prompt() (async).
 * Must be called before the fetch so descriptions are ready.
 */
export async function convertToolsForOllama(tools: Tools): Promise<OllamaTool[]> {
  const results = await Promise.all(
    tools.map(async (tool) => {
      const parameters = getToolParameters(tool)

      let description = ''
      try {
        description = await tool.prompt({
          getToolPermissionContext: async () => ({} as any),
          tools,
          agents: [],
        })
      } catch {
        // prompt() may fail without real context — fall back
      }

      if (!description) {
        description = (tool as any).searchHint || tool.name
      }

      return {
        type: 'function' as const,
        function: {
          name: tool.name,
          description,
          parameters,
        },
      }
    }),
  )
  return results
}

// ---------------------------------------------------------------------------
// Streaming NDJSON parser for Ollama native /api/chat
// ---------------------------------------------------------------------------

async function* parseNDJSONStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): AsyncGenerator<OllamaChatChunk> {
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        yield JSON.parse(trimmed) as OllamaChatChunk
      } catch {
        // Skip malformed JSON lines
      }
    }
  }
  // Flush remaining buffer
  if (buffer.trim()) {
    try {
      yield JSON.parse(buffer.trim()) as OllamaChatChunk
    } catch {
      // ignore
    }
  }
}

// ---------------------------------------------------------------------------
// Main query function for Ollama (native /api/chat)
// ---------------------------------------------------------------------------

export async function* queryModelOllama(
  messages: Message[],
  systemPrompt: SystemPrompt,
  tools: Tools,
  signal: AbortSignal,
  options: Options,
): AsyncGenerator<StreamEvent | AssistantMessage | SystemAPIErrorMessage, void> {
  const baseUrl = getOllamaBaseUrl()
  const model = getOllamaModel()
  const numCtx = getOllamaNumCtx()
  const maxTokens = getOllamaMaxTokens()
  const ollamaMessages = convertMessagesForOllama(messages, systemPrompt, tools)
  const ollamaTools = tools.length > 0 ? await convertToolsForOllama(tools) : undefined

  logForDebugging(`[Ollama] POST ${baseUrl}/api/chat model=${model} messages=${ollamaMessages.length} tools=${ollamaTools?.length ?? 0} num_ctx=${numCtx} num_predict=${maxTokens}`)

  const body: Record<string, unknown> = {
    model,
    messages: ollamaMessages,
    stream: true,
    options: {
      num_ctx: numCtx,
      num_predict: maxTokens,
      temperature: options.temperatureOverride ?? 0,
    },
  }
  if (ollamaTools && ollamaTools.length > 0) {
    body.tools = ollamaTools
  }

  let response: Response
  try {
    response = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : String(err)
    logForDebugging(`[Ollama] Connection error: ${errMsg}`)
    yield {
      type: 'system',
      uuid: randomUUID(),
      message: {
        content: `Ollama connection error: ${errMsg}. Is Ollama running at ${baseUrl}?`,
      },
    } as SystemAPIErrorMessage
    return
  }

  if (!response.ok) {
    const errText = await response.text().catch(() => 'unknown error')
    logForDebugging(`[Ollama] HTTP ${response.status}: ${errText}`)
    yield {
      type: 'system',
      uuid: randomUUID(),
      message: {
        content: `Ollama error (HTTP ${response.status}): ${errText}`,
      },
    } as SystemAPIErrorMessage
    return
  }

  if (!response.body) {
    yield {
      type: 'system',
      uuid: randomUUID(),
      message: { content: 'Ollama returned empty response body' },
    } as SystemAPIErrorMessage
    return
  }

  let fullText = ''
  const toolCalls: { id: string; name: string; arguments: Record<string, unknown> }[] = []
  let inputTokens = 0
  let outputTokens = 0
  let doneReason: string | null = null
  const messageId = randomUUID()
  const start = Date.now()

  yield {
    type: 'stream_event',
    event: {
      type: 'message_start',
      message: {
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    },
  }

  const reader = response.body.getReader()
  let firstChunk = true

  for await (const chunk of parseNDJSONStream(reader)) {
    if (firstChunk) {
      logForDebugging(`[Ollama] First chunk received in ${Date.now() - start}ms`)
      firstChunk = false
    }

    const msg = chunk.message
    if (msg?.content) {
      fullText += msg.content
      yield {
        type: 'stream_event',
        event: {
          type: 'content_block_delta',
          delta: { type: 'text_delta', text: msg.content },
        },
      }
    }

    // Tool calls arrive in the final message (done: true)
    if (msg?.tool_calls) {
      for (const tc of msg.tool_calls) {
        toolCalls.push({
          id: `call_${randomUUID()}`,
          name: tc.function.name,
          arguments: tc.function.arguments,
        })
      }
    }

    if (chunk.done) {
      doneReason = chunk.done_reason ?? 'stop'
      inputTokens = chunk.prompt_eval_count ?? 0
      outputTokens = chunk.eval_count ?? 0
      logForDebugging(`[Ollama] Done: ${inputTokens} input tokens, ${outputTokens} output tokens, reason=${doneReason}`)
    }
  }

  logForDebugging(`[Ollama] Stream complete: ${fullText.length} chars, ${toolCalls.length} tool calls, reason=${doneReason}`)

  // Build AssistantMessage content blocks in Anthropic format
  const contentBlocks: BetaContentBlock[] = []

  if (fullText) {
    contentBlocks.push({
      type: 'text',
      text: fullText,
    } as BetaContentBlock)
  }

  for (const tc of toolCalls) {
    contentBlocks.push({
      type: 'tool_use',
      id: tc.id,
      name: tc.name,
      input: tc.arguments,
    } as unknown as BetaContentBlock)
  }

  if (contentBlocks.length === 0) {
    contentBlocks.push({
      type: 'text',
      text: '(empty response)',
    } as BetaContentBlock)
  }

  const normalizedContent = normalizeContentFromAPI(
    contentBlocks,
    tools,
    options.agentId,
  ) as MessageContent

  const stopReason = toolCalls.length > 0 ? 'tool_use' : 'end_turn'

  const assistantMsg: AssistantMessage = {
    type: 'assistant',
    uuid: messageId,
    timestamp: new Date().toISOString(),
    message: {
      id: `msg_ollama_${messageId}`,
      role: 'assistant',
      content: normalizedContent,
      model,
      stop_reason: stopReason,
      usage: {
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    },
  }

  yield assistantMsg

  yield {
    type: 'stream_event',
    event: { type: 'message_stop' },
  }
}
