/**
 * Ollama adapter for Claude Code.
 *
 * Converts between the internal Anthropic-style message format and
 * OpenAI-compatible chat completions used by Ollama's /v1/chat/completions endpoint.
 * Handles streaming, tool calling, and message normalization.
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
// OpenAI-compatible types (subset needed for Ollama)
// ---------------------------------------------------------------------------

interface OAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | null
  tool_calls?: OAIToolCall[]
  tool_call_id?: string
  name?: string
}

interface OAIToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

interface OAITool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

interface OAIStreamDelta {
  role?: string
  content?: string | null
  tool_calls?: {
    index: number
    id?: string
    type?: string
    function?: { name?: string; arguments?: string }
  }[]
}

interface OAIStreamChoice {
  index: number
  delta: OAIStreamDelta
  finish_reason: string | null
}

interface OAIStreamChunk {
  id: string
  object: string
  model: string
  choices: OAIStreamChoice[]
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
}

// ---------------------------------------------------------------------------
// Message conversion: Internal (Anthropic-style) → OpenAI chat format
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
 * return them as separate OAI "tool" role messages, plus the remaining text.
 */
function extractToolResults(msg: Message): OAIMessage[] {
  const results: OAIMessage[] = []
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
        tool_call_id: String(block.tool_use_id || ''),
        content: resultContent,
      })
    } else if (block.type === 'text' && block.text) {
      textParts.push(String(block.text))
    }
  }

  const oai: OAIMessage[] = []
  if (results.length > 0) {
    oai.push(...results)
  }
  if (textParts.length > 0) {
    oai.push({ role: 'user', content: textParts.join('\n') })
  }
  if (oai.length === 0) {
    oai.push({ role: 'user', content: '' })
  }
  return oai
}

/**
 * Convert an assistant message with potential tool_use blocks to OAI format.
 */
function convertAssistantMessage(msg: Message): OAIMessage {
  const oai: OAIMessage = { role: 'assistant', content: null }
  const textParts: string[] = []
  const toolCalls: OAIToolCall[] = []

  if (!msg.message?.content || typeof msg.message.content === 'string') {
    oai.content = contentToString(msg.message?.content)
    return oai
  }

  for (const block of msg.message.content as Array<Record<string, unknown>>) {
    if (block.type === 'text' && block.text) {
      textParts.push(String(block.text))
    } else if (block.type === 'tool_use') {
      toolCalls.push({
        id: String(block.id || randomUUID()),
        type: 'function',
        function: {
          name: String(block.name || ''),
          arguments:
            typeof block.input === 'string'
              ? block.input
              : JSON.stringify(block.input ?? {}),
        },
      })
    }
  }

  if (textParts.length > 0) oai.content = textParts.join('\n')
  if (toolCalls.length > 0) oai.tool_calls = toolCalls
  return oai
}

export function convertMessagesForOllama(
  messages: Message[],
  systemPrompt: SystemPrompt,
  tools: Tools,
): OAIMessage[] {
  const oaiMessages: OAIMessage[] = []

  const sysText = systemPromptToString(systemPrompt)
  if (sysText) {
    oaiMessages.push({ role: 'system', content: sysText })
  }

  const normalized = normalizeMessagesForAPI(messages, tools)
  for (const msg of normalized) {
    if (msg.type === 'user' || msg.message?.role === 'user') {
      oaiMessages.push(...extractToolResults(msg))
    } else if (msg.type === 'assistant' || msg.message?.role === 'assistant') {
      oaiMessages.push(convertAssistantMessage(msg))
    }
  }

  return oaiMessages
}

// ---------------------------------------------------------------------------
// Tool conversion: Internal (Anthropic-style) → OpenAI function format
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
export async function convertToolsForOllama(tools: Tools): Promise<OAITool[]> {
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
// Streaming SSE parser
// ---------------------------------------------------------------------------

async function* parseSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): AsyncGenerator<OAIStreamChunk> {
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
      if (!trimmed || trimmed.startsWith(':')) continue
      if (trimmed === 'data: [DONE]') return
      if (trimmed.startsWith('data: ')) {
        try {
          yield JSON.parse(trimmed.slice(6)) as OAIStreamChunk
        } catch {
          // Skip malformed JSON chunks
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Main query function for Ollama
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
  const oaiMessages = convertMessagesForOllama(messages, systemPrompt, tools)
  const oaiTools = tools.length > 0 ? await convertToolsForOllama(tools) : undefined

  logForDebugging(`[Ollama] Sending request to ${baseUrl}/v1/chat/completions model=${model} messages=${oaiMessages.length} tools=${oaiTools?.length ?? 0}`)

  const body: Record<string, unknown> = {
    model,
    messages: oaiMessages,
    stream: true,
    temperature: options.temperatureOverride ?? 0,
  }
  if (oaiTools && oaiTools.length > 0) {
    body.tools = oaiTools
  }

  let response: Response
  try {
    response = await fetch(`${baseUrl}/v1/chat/completions`, {
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

  // Accumulate the full response
  let fullText = ''
  const toolCallAccum = new Map<
    number,
    { id: string; name: string; arguments: string }
  >()
  let finishReason: string | null = null
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

  for await (const chunk of parseSSEStream(reader)) {
    if (!chunk.choices?.[0]) continue
    const choice = chunk.choices[0]
    const delta = choice.delta

    if (firstChunk) {
      logForDebugging(`[Ollama] First chunk received in ${Date.now() - start}ms`)
      firstChunk = false
    }

    // Accumulate text content
    if (delta.content) {
      fullText += delta.content
      yield {
        type: 'stream_event',
        event: {
          type: 'content_block_delta',
          delta: { type: 'text_delta', text: delta.content },
        },
      }
    }

    // Accumulate tool calls
    if (delta.tool_calls) {
      for (const tc of delta.tool_calls) {
        const existing = toolCallAccum.get(tc.index)
        if (!existing) {
          toolCallAccum.set(tc.index, {
            id: tc.id || `call_${randomUUID()}`,
            name: tc.function?.name || '',
            arguments: tc.function?.arguments || '',
          })
        } else {
          if (tc.function?.name) existing.name = tc.function.name
          if (tc.function?.arguments) existing.arguments += tc.function.arguments
        }
      }
    }

    if (choice.finish_reason) {
      finishReason = choice.finish_reason
    }
  }

  logForDebugging(`[Ollama] Stream complete: ${fullText.length} chars, ${toolCallAccum.size} tool calls, finish=${finishReason}`)

  // Build the AssistantMessage content blocks in Anthropic format
  const contentBlocks: BetaContentBlock[] = []

  if (fullText) {
    contentBlocks.push({
      type: 'text',
      text: fullText,
    } as BetaContentBlock)
  }

  for (const [, tc] of toolCallAccum) {
    let parsedInput: Record<string, unknown> = {}
    try {
      parsedInput = JSON.parse(tc.arguments)
    } catch {
      parsedInput = {}
    }
    contentBlocks.push({
      type: 'tool_use',
      id: tc.id,
      name: tc.name,
      input: parsedInput,
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

  const stopReason =
    finishReason === 'tool_calls' || finishReason === 'function_call'
      ? 'tool_use'
      : 'end_turn'

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
        input_tokens: 0,
        output_tokens: 0,
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
