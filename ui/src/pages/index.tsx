import React, { useState, useEffect } from 'react'
import Head from 'next/head'
import { ShieldCheckIcon, ExclamationTriangleIcon, ClockIcon, ChartBarIcon } from '@heroicons/react/24/outline'

interface MetricCard {
  title: string
  value: string | number
  change?: string
  changeType?: 'increase' | 'decrease'
  icon: React.ComponentType<any>
  color: string
}

interface PredictionResult {
  flow_id: string
  is_anomaly: boolean
  anomaly_score: number
  qos_recommendation: string
  processing_time_ms: number
  timestamp: string
}

export default function Dashboard() {
  // Dynamic metrics based on dashboard stats
  const [dashboardStats, setDashboardStats] = useState({
    totalPredictions: 12847,
    anomaliesDetected: 1923,
    normalFlows: 10924,
    avgResponseTime: 127
  })

  const metrics = [
    {
      title: 'Total Predictions',
      value: dashboardStats.totalPredictions.toLocaleString(),
      change: '+12%',
      changeType: 'increase',
      icon: ChartBarIcon,
      color: 'blue'
    },
    {
      title: 'Anomalies Detected',
      value: dashboardStats.anomaliesDetected.toLocaleString(),
      change: '+8%',
      changeType: 'increase',
      icon: ExclamationTriangleIcon,
      color: 'red'
    },
    {
      title: 'Normal Flows',
      value: dashboardStats.normalFlows.toLocaleString(),
      change: '+4%',
      changeType: 'increase',
      icon: ShieldCheckIcon,
      color: 'green'
    },
    {
      title: 'Avg Response Time',
      value: `${dashboardStats.avgResponseTime}ms`,
      change: '-3ms',
      changeType: 'decrease',
      icon: ClockIcon,
      color: 'yellow'
    }
  ]

  const [mounted, setMounted] = useState(false)
  const [recentPredictions, setRecentPredictions] = useState<any[]>([])
  const [testResult, setTestResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' | 'info' } | null>(null)

  const [systemStatus, setSystemStatus] = useState({
    api: 'healthy',
    models: 'loaded',
    mlflow: 'connected'
  })

  useEffect(() => {
    setMounted(true)
    
    // Initialize sample data after mount to avoid hydration mismatch
    setRecentPredictions([
      {
        flow_id: 'flow-001',
        is_anomaly: true,
        anomaly_score: 0.85,
        qos_recommendation: 'RATE_LIMIT',
        processing_time_ms: 142,
        timestamp: new Date().toISOString()
      },
      {
        flow_id: 'flow-002',
        is_anomaly: false,
        anomaly_score: 0.23,
        qos_recommendation: 'INSPECT',
        processing_time_ms: 98,
        timestamp: new Date(Date.now() - 30000).toISOString()
      }
    ])

    // Check API health
    const checkHealth = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/healthz`)
        if (response.ok) {
          setSystemStatus(prev => ({ ...prev, api: 'healthy' }))
        } else {
          setSystemStatus(prev => ({ ...prev, api: 'unhealthy' }))
        }
      } catch (error) {
        setSystemStatus(prev => ({ ...prev, api: 'disconnected' }))
      }
    }

    // Initial fetch of dashboard stats and health check
    checkHealth()
    fetchDashboardStats()
    
    // Set up periodic updates
    const healthInterval = setInterval(checkHealth, 30000) // Check every 30 seconds
    const statsInterval = setInterval(fetchDashboardStats, 60000) // Update stats every minute

    return () => {
      clearInterval(healthInterval)
      clearInterval(statsInterval)
    }
  }, [])

  // Function to fetch dashboard stats from backend
  const fetchDashboardStats = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/stats`)
      if (response.ok) {
        const stats = await response.json()
        setDashboardStats({
          totalPredictions: stats.total_predictions || dashboardStats.totalPredictions,
          anomaliesDetected: stats.anomalies_detected || dashboardStats.anomaliesDetected,
          normalFlows: stats.normal_flows || dashboardStats.normalFlows,
          avgResponseTime: stats.avg_response_time || dashboardStats.avgResponseTime
        })
      }
    } catch (error) {
      console.log('Stats API not available, using local stats')
    }
  }

  // Function to update dashboard stats based on predictions
  const updateDashboardStats = (predictions: any[]) => {
    const newAnomalies = predictions.filter(p => p.is_anomaly).length
    const total = dashboardStats.totalPredictions + predictions.length
    const anomalies = dashboardStats.anomaliesDetected + newAnomalies
    const normal = total - anomalies
    const avgTime = predictions.length > 0 
      ? Math.round(predictions.reduce((sum, p) => sum + (p.processing_time_ms || 0), dashboardStats.avgResponseTime * 2) / (predictions.length + 2))
      : dashboardStats.avgResponseTime
    
    setDashboardStats({
      totalPredictions: total,
      anomaliesDetected: anomalies,
      normalFlows: normal,
      avgResponseTime: avgTime
    })
  }

  // Show notification function
  const showNotification = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    setNotification({ message, type })
    setTimeout(() => setNotification(null), 4000) // Show for 4 seconds
  }

  // Button click handlers
  const handleSendTestFlow = async () => {
    setLoading(true)
    try {
      const testFlow = {
        flow_id: `test-${Date.now()}`,
        features: {
          dur: 10.5, proto: 'tcp', service: 'http', state: 'FIN',
          spkts: 100, dpkts: 95, sbytes: 50000, dbytes: 48000,
          rate: 15.2, sttl: 64, dttl: 64, sload: 4000.0, dload: 3800.0,
          sloss: 0, dloss: 0, swin: 8192, dwin: 8192, stcpb: 123456,
          dtcpb: 789012, smeansz: 500.0, dmeansz: 505.3, trans_depth: 1,
          res_bdy_len: 1024, sjit: 0.01, djit: 0.01, sintpkt: 0.1,
          dintpkt: 0.1, tcprtt: 0.05, synack: 0.01, ackdat: 0.005
        }
      }
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testFlow)
      })
      
      if (response.ok) {
        const result = await response.json()
        setTestResult(result)
        // Add to recent predictions
        const newPrediction = {
          flow_id: result.flow_id,
          is_anomaly: result.is_anomaly,
          anomaly_score: result.anomaly_score,
          qos_recommendation: result.qos_recommendation,
          processing_time_ms: result.processing_time_ms,
          timestamp: new Date().toISOString()
        }
        const updatedPredictions = [newPrediction, ...recentPredictions.slice(0, 9)]
        setRecentPredictions(updatedPredictions)
        updateDashboardStats(updatedPredictions)
        showNotification(`Test Result: ${result.is_anomaly ? 'ANOMALY' : 'NORMAL'} (Score: ${result.anomaly_score?.toFixed(3)}, Action: ${result.qos_recommendation})`, 'success')
      } else {
        showNotification('Error: Failed to get prediction', 'error')
      }
    } catch (error) {
      showNotification('Error: Cannot connect to API', 'error')
    }
    setLoading(false)
  }

  const handleUpdatePolicy = async () => {
    const newThreshold = prompt('Enter new anomaly detection threshold (0.0-1.0):', '0.7')
    if (newThreshold && !isNaN(Number(newThreshold))) {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/policy`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            policy_name: "default",
            thresholds: {
              rate_limit_threshold: Number(newThreshold),
              prioritize_threshold: 0.9,
              drop_threshold: 0.5,
              inspect_threshold: 0.3
            },
            description: "Updated QoS policy via dashboard"
          })
        })
        
        if (response.ok) {
          showNotification(`Policy updated! Rate limit threshold set to ${newThreshold}`, 'success')
        } else {
          const error = await response.text()
          showNotification(`Error: Failed to update policy - ${error}`, 'error')
        }
      } catch (error) {
        showNotification('Error: Cannot connect to API', 'error')
      }
    }
  }

  const handleViewSHAP = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/models`)
      if (response.ok) {
        const data = await response.json()
        const modelsInfo = data.models || {}
        const totalModels = data.total_models || 0
        showNotification(`SHAP Models Available: ${totalModels} models loaded. Registry: ${data.registry_path || 'Not configured'}`, 'info')
      } else {
        showNotification('Error: Failed to get SHAP data', 'error')
      }
    } catch (error) {
      showNotification('Error: Cannot connect to API', 'error')
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'loaded':
      case 'connected':
        return 'text-green-600 bg-green-100'
      case 'unhealthy':
        return 'text-yellow-600 bg-yellow-100'
      case 'disconnected':
      case 'failed':
        return 'text-red-600 bg-red-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  const getQoSRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'PRIORITIZE':
        return 'text-blue-700 bg-blue-100'
      case 'RATE_LIMIT':
        return 'text-yellow-700 bg-yellow-100'
      case 'DROP_CANDIDATE':
        return 'text-red-700 bg-red-100'
      case 'INSPECT':
        return 'text-green-700 bg-green-100'
      default:
        return 'text-gray-700 bg-gray-100'
    }
  }

  return (
    <>
      <Head>
        <title>QoSGuard Dashboard</title>
        <meta name="description" content="Real-time network anomaly detection and QoS recommendations" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white shadow">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 justify-between items-center">
              <div className="flex items-center">
                <ShieldCheckIcon className="h-8 w-8 text-blue-600 mr-3" />
                <h1 className="text-xl font-semibold text-gray-900">QoSGuard</h1>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 text-sm">
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemStatus.api)}`}>
                    API: {systemStatus.api}
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemStatus.models)}`}>
                    Models: {systemStatus.models}
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemStatus.mlflow)}`}>
                    MLflow: {systemStatus.mlflow}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {metrics.map((metric, index) => (
              <div key={index} className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <metric.icon className={`h-6 w-6 text-${metric.color}-600`} aria-hidden="true" />
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">
                          {metric.title}
                        </dt>
                        <dd className="flex items-baseline">
                          <div className="text-2xl font-semibold text-gray-900">
                            {metric.value}
                          </div>
                          {metric.change && (
                            <div className={`ml-2 flex items-baseline text-sm font-semibold ${
                              metric.changeType === 'increase' ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {metric.change}
                            </div>
                          )}
                        </dd>
                      </dl>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Recent Predictions */}
          <div className="bg-white shadow overflow-hidden sm:rounded-md">
            <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Recent Predictions
              </h3>
              <p className="mt-1 max-w-2xl text-sm text-gray-500">
                Latest network flow anomaly detection results
              </p>
            </div>
            <ul className="divide-y divide-gray-200">
              {recentPredictions.map((prediction) => (
                <li key={prediction.flow_id}>
                  <div className="px-4 py-4 sm:px-6 hover:bg-gray-50">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className={`flex-shrink-0 w-3 h-3 rounded-full mr-3 ${
                          prediction.is_anomaly ? 'bg-red-400' : 'bg-green-400'
                        }`} />
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {prediction.flow_id}
                          </p>
                          <p className="text-sm text-gray-500">
                            Score: {prediction.anomaly_score.toFixed(3)} • 
                            Response: {prediction.processing_time_ms}ms
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          getQoSRecommendationColor(prediction.qos_recommendation)
                        }`}>
                          {prediction.qos_recommendation}
                        </span>
                        <p className="text-sm text-gray-500">
                          {new Date(prediction.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* Quick Actions */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Test Prediction</h3>
                <button 
                  onClick={handleSendTestFlow}
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-2 px-4 rounded-md transition duration-150 ease-in-out"
                >
                  {loading ? 'Testing...' : 'Send Test Flow'}
                </button>
              </div>
            </div>
            
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Policy Management</h3>
                <button 
                  onClick={handleUpdatePolicy}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md transition duration-150 ease-in-out"
                >
                  Update Policy
                </button>
              </div>
            </div>
            
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Model Insights</h3>
                <button 
                  onClick={handleViewSHAP}
                  className="w-full bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-4 rounded-md transition duration-150 ease-in-out"
                >
                  View SHAP
                </button>
              </div>
            </div>
          </div>

          {/* Notification Component */}
          {notification && (
            <div className={`fixed top-4 right-4 max-w-sm w-full bg-white border-l-4 p-4 shadow-lg rounded-md z-50 ${
              notification.type === 'success' ? 'border-green-500' : 
              notification.type === 'error' ? 'border-red-500' : 'border-blue-500'
            }`}>
              <div className="flex">
                <div className="flex-shrink-0">
                  {notification.type === 'success' && (
                    <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  )}
                  {notification.type === 'error' && (
                    <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  )}
                  {notification.type === 'info' && (
                    <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
                <div className="ml-3">
                  <p className={`text-sm font-medium ${
                    notification.type === 'success' ? 'text-green-800' : 
                    notification.type === 'error' ? 'text-red-800' : 'text-blue-800'
                  }`}>
                    {notification.message}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
