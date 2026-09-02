<script setup>
import { ref, nextTick, computed } from 'vue'
import { chat, errorMessage } from '../api'

const userId = ref('demo-user')
const sessionId = ref('web-session')
const mode = ref('react') // react | plan
const input = ref('')
const loading = ref(false)
const error = ref('')
const messages = ref([])

const listEl = ref(null)

const suggestions = [
  '现在车速多少?',
  '有点冷',
  '把温度调到24度',
  '帮我开空调',
  '导航去公司',
  '播放周杰伦的音乐',
  '外面天气怎么样?',
]

async function scrollToBottom() {
  await nextTick()
  if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight
}

async function send(text) {
  const content = (text ?? input.value).trim()
  if (!content || loading.value) return
  input.value = ''
  error.value = ''
  messages.value.push({ role: 'user', content })
  loading.value = true
  await scrollToBottom()
  try {
    const res = await chat(content, {
      userId: userId.value,
      sessionId: sessionId.value,
      mode: mode.value,
    })
    messages.value.push({
      role: 'assistant',
      content: res.response || '(无回复)',
      toolCalls: res.tool_calls || [],
      requestId: res.request_id,
      latencyMs: res.latency_ms,
    })
  } catch (err) {
    error.value = errorMessage(err)
    messages.value.push({ role: 'assistant', content: `出错: ${error.value}`, error: true })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

function onEnter(e) {
  if (e.shiftKey) return
  e.preventDefault()
  send()
}

function clearHistory() {
  messages.value = []
  error.value = ''
}

const placeholder = computed(() =>
  mode.value === 'plan'
    ? '输入多步任务,例如:导航去公司并把温度调到24度'
    : '和车载助手对话,例如:现在车速多少?',
)
</script>

<template>
  <section class="panel">
    <div class="toolbar">
      <label class="field">
        <span>用户</span>
        <input v-model="userId" />
      </label>
      <label class="field">
        <span>会话</span>
        <input v-model="sessionId" />
      </label>
      <div class="mode-switch">
        <button :class="{ active: mode === 'react' }" @click="mode = 'react'">单轮 ReAct</button>
        <button :class="{ active: mode === 'plan' }" @click="mode = 'plan'">多步 Plan</button>
      </div>
      <button class="ghost" @click="clearHistory">清空</button>
    </div>

    <div ref="listEl" class="messages scrollbar">
      <div v-if="messages.length === 0" class="empty">
        <p>👋 你好!我是 AutoAgent 车载智能助手。</p>
        <p class="dim">试试下面的快捷指令:</p>
      </div>

      <div v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
        <div class="bubble">
          <div class="text">{{ m.content }}</div>
          <template v-if="m.toolCalls && m.toolCalls.length">
            <div class="tools">
              <div v-for="(tc, j) in m.toolCalls" :key="j" class="tool">
                <span class="tool-name">🔧 {{ tc.name }}</span>
                <code class="tool-args">{{ JSON.stringify(tc.arguments) }}</code>
              </div>
            </div>
          </template>
          <div class="meta">
            <span v-if="m.requestId" class="meta-item">{{ m.requestId }}</span>
            <span v-if="m.latencyMs != null" class="meta-item">{{ m.latencyMs.toFixed(0) }} ms</span>
          </div>
        </div>
      </div>

      <div v-if="loading" class="msg assistant">
        <div class="bubble typing">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>

    <div v-if="error" class="error-bar">{{ error }}</div>

    <div class="composer">
      <textarea
        v-model="input"
        :placeholder="placeholder"
        rows="2"
        @keydown.enter="onEnter"
      ></textarea>
      <button class="primary send" :disabled="loading || !input.trim()" @click="send()">
        {{ loading ? '处理中...' : '发送' }}
      </button>
    </div>

    <div class="suggestions">
      <button v-for="s in suggestions" :key="s" class="chip" @click="send(s)">{{ s }}</button>
    </div>
  </section>
</template>

<style scoped>
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  display: flex;
  flex-direction: column;
  height: 70vh;
  min-height: 520px;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}
.field {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-dim);
}
.field input {
  width: 130px;
  padding: 6px 8px;
  font-size: 12px;
}
.mode-switch {
  display: flex;
  background: var(--bg);
  border-radius: 8px;
  padding: 3px;
  gap: 2px;
}
.mode-switch button {
  background: transparent;
  padding: 5px 12px;
  font-size: 12px;
  color: var(--text-dim);
}
.mode-switch button.active {
  background: var(--panel-2);
  color: var(--text);
}
button.ghost {
  margin-left: auto;
  border: 1px solid var(--border);
  padding: 6px 12px;
  font-size: 12px;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.empty {
  text-align: center;
  color: var(--text-dim);
  margin: auto;
}
.empty .dim {
  font-size: 12px;
  margin-top: 4px;
}
.msg {
  display: flex;
}
.msg.user {
  justify-content: flex-end;
}
.bubble {
  max-width: 78%;
  padding: 10px 14px;
  border-radius: 14px;
  background: var(--panel-2);
}
.msg.user .bubble {
  background: var(--accent);
  color: #0a1419;
}
.text {
  white-space: pre-wrap;
  word-break: break-word;
}
.tools {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.tool {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(0, 0, 0, 0.25);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
}
.tool-name {
  font-weight: 600;
  color: var(--accent);
  white-space: nowrap;
}
.tool-args {
  color: var(--text-dim);
  font-size: 11px;
}
.meta {
  margin-top: 6px;
  display: flex;
  gap: 10px;
  font-size: 10px;
  color: var(--text-dim);
  opacity: 0.7;
}
.typing {
  display: flex;
  gap: 4px;
  align-items: center;
}
.typing span {
  width: 7px;
  height: 7px;
  background: var(--text-dim);
  border-radius: 50%;
  animation: blink 1.2s infinite both;
}
.typing span:nth-child(2) {
  animation-delay: 0.2s;
}
.typing span:nth-child(3) {
  animation-delay: 0.4s;
}
@keyframes blink {
  0%, 80%, 100% { opacity: 0.2; }
  40% { opacity: 1; }
}
.error-bar {
  padding: 8px 14px;
  background: rgba(245, 101, 101, 0.12);
  color: var(--danger);
  font-size: 12px;
}
.composer {
  display: flex;
  gap: 8px;
  padding: 12px 14px;
  border-top: 1px solid var(--border);
}
.composer textarea {
  flex: 1;
  resize: none;
}
.send {
  align-self: stretch;
  padding: 0 20px;
}
.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 0 14px 14px;
}
.chip {
  background: var(--bg);
  border: 1px solid var(--border);
  font-size: 12px;
  padding: 5px 10px;
  color: var(--text-dim);
}
.chip:hover {
  border-color: var(--accent);
  color: var(--text);
}
</style>
