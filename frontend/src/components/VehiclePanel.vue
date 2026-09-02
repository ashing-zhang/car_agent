<script setup>
import { ref, onMounted } from 'vue'
import { getVehicleStatus, setTemperature, errorMessage } from '../api'

const status = ref(null)
const loading = ref(false)
const error = ref('')
const targetTemp = ref(24)
const tempMsg = ref('')

async function refresh() {
  loading.value = true
  error.value = ''
  try {
    status.value = await getVehicleStatus()
  } catch (err) {
    error.value = errorMessage(err)
  } finally {
    loading.value = false
  }
}

async function applyTemp() {
  tempMsg.value = ''
  error.value = ''
  try {
    const res = await setTemperature(targetTemp.value)
    tempMsg.value = res.output || res.error || '已设置'
    await refresh()
  } catch (err) {
    error.value = errorMessage(err)
  }
}

onMounted(refresh)
</script>

<template>
  <section class="panel">
    <div class="header-row">
      <h2>车辆状态</h2>
      <button class="primary" :disabled="loading" @click="refresh">
        {{ loading ? '加载中...' : '刷新状态' }}
      </button>
    </div>

    <div v-if="error" class="error">{{ error }}</div>

    <div v-if="status" class="status-card">
      <div class="success badge ok" v-if="status.success">{{ status.tool_name }}</div>
      <pre class="output">{{ status.output }}</pre>
      <div v-if="status.error" class="err-detail">{{ status.error }}</div>
    </div>
    <div v-else-if="!loading" class="placeholder">暂无数据,点击刷新</div>

    <div class="control-card">
      <h3>温度控制</h3>
      <div class="control-row">
        <label>目标温度</label>
        <input v-model.number="targetTemp" type="number" min="0" max="50" step="1" />
        <span class="unit">℃</span>
        <button class="primary" @click="applyTemp">应用</button>
      </div>
      <div v-if="tempMsg" class="temp-msg">{{ tempMsg }}</div>
      <p class="hint">安全范围 18-30℃(超出将被策略拒绝)</p>
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
.status-card,
.control-card {
  background: var(--panel-2);
  border-radius: 10px;
  padding: 14px;
}
.output {
  margin: 10px 0 0;
  white-space: pre-wrap;
  font-family: 'SF Mono', 'Consolas', monospace;
  font-size: 13px;
  color: var(--text);
}
.err-detail {
  color: var(--danger);
  font-size: 12px;
  margin-top: 8px;
}
.placeholder {
  color: var(--text-dim);
  text-align: center;
  padding: 24px;
}
.control-card h3 {
  margin: 0 0 12px;
  font-size: 14px;
  color: var(--accent);
}
.control-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.control-row input {
  width: 90px;
}
.unit {
  color: var(--text-dim);
}
.temp-msg {
  margin-top: 10px;
  color: var(--ok);
  font-size: 12px;
}
.hint {
  margin: 10px 0 0;
  font-size: 11px;
  color: var(--text-dim);
}
.error {
  padding: 10px 12px;
  background: rgba(245, 101, 101, 0.12);
  color: var(--danger);
  border-radius: 8px;
  font-size: 12px;
}
.success {
  margin-bottom: 8px;
}
</style>
