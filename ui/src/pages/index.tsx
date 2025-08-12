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
  const [metrics, setMetrics] = useState<MetricCard[]>([
    {
      title: 'Total Predictions',
      value: '12,847',
      change: '+12%',
      changeType: 'increase',
      icon: ChartBarIcon,
      color: 'blue'
    },
    {
      title: 'Anomalies Detected',
      value: '1,923',
      change: '+8%',
      changeType: 'increase',
      icon: ExclamationTriangleIcon,
      color: 'red'
    },
    {
      title: 'Normal Flows',
      value: '10,924',
      change: '+4%',
      changeType: 'increase',
      icon: ShieldCheckIcon,
      color: 'green'
    },
    {
      title: 'Avg Response Time',
      value: '127ms',
      change: '-3ms',
      changeType: 'decrease',
      icon: ClockIcon,
      color: 'yellow'
    }
  ])

  const [recentPredictions, setRecentPredictions] = useState<PredictionResult[]>([
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

  const [systemStatus, setSystemStatus] = useState({
    api: 'healthy',
    models: 'loaded',
    mlflow: 'connected'
  })

  useEffect(() => {
    // Check API health
    const checkHealth = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/healthz`)
        if (response.ok) {
          setSystemStatus(prev => ({ ...prev, api: 'healthy' }))
        } else {
          setSystemStatus(prev => ({ ...prev, api: 'unhealthy' }))
        }
      } catch (error) {
        setSystemStatus(prev => ({ ...prev, api: 'disconnected' }))
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Check every 30 seconds

    return () => clearInterval(interval)
  }, [])

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
                <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition duration-150 ease-in-out">
                  Send Test Flow
                </button>
              </div>
            </div>
            
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Policy Management</h3>
                <button className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md transition duration-150 ease-in-out">
                  Update Policy
                </button>
              </div>
            </div>
            
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Model Insights</h3>
                <button className="w-full bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-4 rounded-md transition duration-150 ease-in-out">
                  View SHAP
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
