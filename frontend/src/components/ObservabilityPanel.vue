<script setup>
import { ref, onMounted, computed } from 'vue'
import { getDashboard, getTraceDetail, errorMessage } from '../api'

const dashboard = ref(null)
const loading = ref(false)
const error = ref('')
const selectedTrace = ref(null)

async function refresh() {
  loading.value = true
  error.value = ''
  try {
    dashboard.value = await getDashboard()
  } catch (err) {
    error.value = errorMessage(err)
  } finally {
    loading.value = false
  }
}

async function showDetail(id) {
  try {
    selectedTrace.value = await getTraceDetail(id)
  } catch (err) {
    error.value = errorMessage(err)
  }
}

const traces = computed(() => dashboard.value?.recent_traces || [])
const metrics = computed(() => dashboard.value?.metrics || { counters: {}, histograms: {} })
const counters = computed(() => metrics.value.counters || {})
const histograms = computed(() => metrics.value.histograms || {})

function fmtTime(ms) {
  if (!ms) return '-'
  return new Date(ms).toLocaleTimeString('zh-CN', { hour12: false })
}

function fmtLatency(ms) {
  if (ms == null) return '-'
  return ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`
}

onMounted(refresh)
</script>

<template>
  <section class="panel">
    <div class="header-row">
      <h2>可观测性仪表盘</h2>
      <button class="primary" :disabled="loading" @click="refresh">
        {{ loading ? '加载中...' : '刷新' }}
      </button>
    </div>

    <div v-if="error" class="error">{{ error }}</div>

    <div class="metrics-grid">
      <div v-for="(samples, name) in counters" :key="'c-' + name" class="metric-card">
        <div class="metric-title">{{ name }}</div>
        <div v-for="(s, i) in samples" :key="i" class="metric-row">
          <span class="metric-label">{{ JSON.stringify(s.labels) }}</span>
          <span class="metric-value">{{ s.value }}</span>
        </div>
        <div v-if="!samples.length" class="metric-row">
          <span class="metric-label">(无样本)</span>
          <span class="metric-value">0</span>
        </div>
      </div>

      <div v-for="(samples, name) in histograms" :key="'h-' + name" class="metric-card">
        <div class="metric-title">{{ name }}</div>
        <div v-for="(s, i) in samples" :key="i" class="metric-row">
          <span class="metric-label">count/sum {{ JSON.stringify(s.labels) }}</span>
          <span class="metric-value">{{ s.total }} / {{ s.sum.toFixed(0) }}</span>
        </div>
        <div v-if="!samples.length" class="metric-row">
          <span class="metric-label">(无样本)</span>
          <span class="metric-value">0</span>
        </div>
      </div>
    </div>

    <div class="traces-section">
      <div class="section-title">最近 Trace ({{ traces.length }})</div>
      <div class="trace-list scrollbar">
        <table>
          <thead>
            <tr>
              <th>request_id</th>
              <th>用户</th>
              <th>输入</th>
              <th>状态</th>
              <th>工具</th>
              <th>延迟</th>
              <th>时间</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="t in traces"
              :key="t.request_id"
              class="trace-row"
              @click="showDetail(t.request_id)"
            >
              <td class="mono">{{ t.request_id.slice(-8) }}</td>
              <td>{{ t.user_id }}</td>
              <td class="ellipsis">{{ t.input }}</td>
              <td>
                <span class="badge" :class="t.success ? 'ok' : 'error'">
                  {{ t.success ? 'ok' : 'fail' }}
                </span>
              </td>
              <td>{{ t.tool_call_count }}</td>
              <td>{{ fmtLatency(t.latency_ms) }}</td>
              <td class="mono">{{ fmtTime(t.created_at_ms) }}</td>
            </tr>
            <tr v-if="!traces.length">
              <td colspan="7" class="empty">暂无 trace,先到"智能助手"发起一次对话</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div v-if="selectedTrace" class="detail-modal" @click.self="selectedTrace = null">
      <div class="modal-card scrollbar">
        <div class="modal-header">
          <h3>Trace 详情</h3>
          <button class="ghost" @click="selectedTrace = null">关闭</button>
        </div>
        <div class="detail-grid">
          <div><span class="dim">request_id</span>{{ selectedTrace.request_id }}</div>
          <div><span class="dim">model</span>{{ selectedTrace.model }}</div>
          <div><span class="dim">latency</span>{{ fmtLatency(selectedTrace.latency_ms) }}</div>
          <div><span class="dim">memory reads/writes</span>{{ selectedTrace.memory_reads }} / {{ selectedTrace.memory_writes }}</div>
        </div>
        <div class="detail-block">
          <div class="dim">输入</div>
          <pre>{{ selectedTrace.input }}</pre>
        </div>
        <div class="detail-block">
          <div class="dim">最终回复</div>
          <pre>{{ selectedTrace.final_response || '(空)' }}</pre>
        </div>
        <div v-if="selectedTrace.tool_calls?.length" class="detail-block">
          <div class="dim">工具调用</div>
          <pre>{{ JSON.stringify(selectedTrace.tool_calls, null, 2) }}</pre>
        </div>
        <div v-if="selectedTrace.spans?.length" class="detail-block">
          <div class="dim">Spans</div>
          <div v-for="s in selectedTrace.spans" :key="s.span_id" class="span-row">
            <span class="mono">{{ s.name }}</span>
            <span class="dim">{{ fmtLatency(s.latency_ms) }}</span>
            <span class="badge" :class="s.status === 'ok' ? 'ok' : 'error'">{{ s.status }}</span>
          </div>
        </div>
        <div v-if="selectedTrace.error" class="err-box">{{ selectedTrace.error }}</div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
h2 {
  margin: 0;
  font-size: 16px;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
}
.metric-card {
  background: var(--panel-2);
  border-radius: 10px;
  padding: 12px;
}
.metric-title {
  font-size: 12px;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 6px;
  word-break: break-all;
}
.metric-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 11px;
  color: var(--text-dim);
  padding: 2px 0;
}
.metric-value {
  color: var(--text);
  font-weight: 600;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
}
.trace-list {
  max-height: 320px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
th {
  text-align: left;
  padding: 8px 10px;
  background: var(--panel-2);
  color: var(--text-dim);
  font-weight: 600;
  position: sticky;
  top: 0;
}
td {
  padding: 7px 10px;
  border-top: 1px solid var(--border);
}
.trace-row {
  cursor: pointer;
}
.trace-row:hover {
  background: var(--panel-2);
}
.mono {
  font-family: 'SF Mono', 'Consolas', monospace;
}
.ellipsis {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.empty {
  text-align: center;
  color: var(--text-dim);
  padding: 20px;
}
.detail-modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
  padding: 20px;
}
.modal-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  max-width: 640px;
  width: 100%;
  max-height: 80vh;
  overflow-y: auto;
  padding: 18px;
}
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
h3 {
  margin: 0;
  font-size: 15px;
}
button.ghost {
  background: transparent;
  border: 1px solid var(--border);
  padding: 4px 12px;
  font-size: 12px;
}
.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 16px;
  font-size: 12px;
  margin-bottom: 14px;
}
.detail-grid div {
  display: flex;
  flex-direction: column;
}
.dim {
  color: var(--text-dim);
  font-size: 10px;
  margin-bottom: 2px;
}
.detail-block {
  margin-bottom: 12px;
}
.detail-block pre {
  margin: 4px 0 0;
  background: var(--bg);
  padding: 10px;
  border-radius: 6px;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.span-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 8px;
  font-size: 12px;
  background: var(--bg);
  border-radius: 6px;
  margin-bottom: 4px;
}
.err-box {
  color: var(--danger);
  background: rgba(245, 101, 101, 0.12);
  padding: 8px;
  border-radius: 6px;
  font-size: 12px;
}
.error {
  padding: 10px 12px;
  background: rgba(245, 101, 101, 0.12);
  color: var(--danger);
  border-radius: 8px;
  font-size: 12px;
}
</style>
