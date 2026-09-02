<script setup>
import { ref, onMounted } from 'vue'
import ChatPanel from './components/ChatPanel.vue'
import VehiclePanel from './components/VehiclePanel.vue'
import ObservabilityPanel from './components/ObservabilityPanel.vue'
import { getHealth } from './api'

const tabs = [
  { key: 'chat', label: '智能助手' },
  { key: 'vehicle', label: '车辆控制' },
  { key: 'observability', label: '可观测性' },
]
const activeTab = ref('chat')

const health = ref({ status: 'unknown' })
async function refreshHealth() {
  try {
    health.value = await getHealth()
  } catch {
    health.value = { status: 'offline' }
  }
}
onMounted(refreshHealth)
</script>

<template>
  <div class="layout">
    <header class="header">
      <div class="brand">
        <span class="logo">🚗</span>
        <div>
          <h1>AutoAgent</h1>
          <p class="sub">多模态个性化车载 Agent</p>
        </div>
      </div>
      <div class="health">
        <span
          class="badge"
          :class="health.status === 'ok' ? 'ok' : health.status === 'offline' ? 'error' : ''"
        >{{ health.status }}</span>
        <button class="ghost" @click="refreshHealth">刷新</button>
      </div>
    </header>

    <nav class="tabs">
      <button
        v-for="t in tabs"
        :key="t.key"
        :class="['tab', { active: activeTab === t.key }]"
        @click="activeTab = t.key"
      >{{ t.label }}</button>
    </nav>

    <main class="content">
      <ChatPanel v-if="activeTab === 'chat'" />
      <VehiclePanel v-else-if="activeTab === 'vehicle'" />
      <ObservabilityPanel v-else />
    </main>
  </div>
</template>

<style scoped>
.layout {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 20px 40px;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.logo {
  font-size: 30px;
}
h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
}
.sub {
  margin: 2px 0 0;
  color: var(--text-dim);
  font-size: 12px;
}
.health {
  display: flex;
  align-items: center;
  gap: 10px;
}
.tabs {
  display: flex;
  gap: 6px;
  margin: 18px 0 0;
  background: var(--panel);
  border-radius: var(--radius);
  padding: 6px;
}
.tab {
  flex: 1;
  background: transparent;
  padding: 10px 14px;
  border-radius: 8px;
  color: var(--text-dim);
  font-weight: 500;
}
.tab.active {
  background: var(--panel-2);
  color: var(--text);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
.content {
  margin-top: 18px;
}
button.ghost {
  background: transparent;
  border: 1px solid var(--border);
  padding: 4px 12px;
  font-size: 12px;
}
</style>
