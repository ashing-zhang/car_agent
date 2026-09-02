// API 客户端 - 封装对后端 FastAPI 的调用
// 开发模式经 Vite 代理直连后端;生产模式同源部署
import axios from 'axios'

const client = axios.create({
  baseURL: import.meta.env.DEV ? '' : '',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

export async function chat(message, { userId = 'demo-user', sessionId = 'web-session', mode = 'react' } = {}) {
  const url = mode === 'plan' ? '/api/v1/agent/plan' : '/api/v1/agent/chat'
  const { data } = await client.post(url, {
    user_id: userId,
    session_id: sessionId,
    message,
  })
  return data
}

export async function getVehicleStatus() {
  const { data } = await client.get('/api/v1/vehicle/status')
  return data
}

export async function setTemperature(temperatureC) {
  const { data } = await client.post('/api/v1/vehicle/temperature', { temperature_c: temperatureC })
  return data
}

export async function getDashboard() {
  const { data } = await client.get('/api/v1/observability/dashboard')
  return data
}

export async function getTraces(limit = 50) {
  const { data } = await client.get('/api/v1/observability/traces', { params: { limit } })
  return data
}

export async function getTraceDetail(requestId) {
  const { data } = await client.get(`/api/v1/observability/traces/${requestId}`)
  return data
}

export async function getMetricsSnapshot() {
  const { data } = await client.get('/api/v1/observability/metrics')
  return data
}

export async function getHealth() {
  const { data } = await client.get('/api/v1/health')
  return data
}

export function errorMessage(err) {
  if (err?.response?.data?.detail) return err.response.data.detail
  if (err?.message) return err.message
  return '请求失败'
}
