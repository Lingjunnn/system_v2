import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface AlarmLog {
  id: number
  timestamp: string
  type: 'warning' | 'critical'
  message: string
  location: string
}

export type SystemStatus = 'normal' | 'warning' | 'critical'

export const useMonitorStore = defineStore('monitor', () => {
  const isFighting = ref(false)
  const fps = ref(0)
  const alarmLogs = ref<AlarmLog[]>([])
  const videoSource = ref<'camera' | 'file' | 'folder' | null>(null)
  const isModelLoaded = ref(false)
  const currentLogId = ref(0)

  const systemStatus = computed<SystemStatus>(() => {
    if (isFighting.value) return 'critical'
    return 'normal'
  })

  const addAlarmLog = (type: 'warning' | 'critical', message: string, location: string = 'Mine Area A') => {
    const now = new Date()
    const timestamp = now.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })

    currentLogId.value++
    const newLog: AlarmLog = {
      id: currentLogId.value,
      timestamp,
      type,
      message,
      location
    }

    alarmLogs.value.unshift(newLog)

    if (alarmLogs.value.length > 100) {
      alarmLogs.value.pop()
    }
  }

  const clearAlarmLogs = () => {
    alarmLogs.value = []
  }

  const setFighting = (fighting: boolean) => {
    isFighting.value = fighting
    if (fighting) {
      addAlarmLog('critical', 'Fighting detected! Immediate attention required.')
    }
  }

  const setFps = (value: number) => {
    fps.value = value
  }

  const setVideoSource = (source: 'camera' | 'file' | 'folder' | null) => {
    videoSource.value = source
  }

  const setModelLoaded = (loaded: boolean) => {
    isModelLoaded.value = loaded
  }

  return {
    isFighting,
    fps,
    alarmLogs,
    videoSource,
    isModelLoaded,
    systemStatus,
    addAlarmLog,
    clearAlarmLogs,
    setFighting,
    setFps,
    setVideoSource,
    setModelLoaded
  }
})
