import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { Image, Box, Play, GitCompare } from 'lucide-react'

export default function Demo() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState('input')

  const tabs = [
    { id: 'input', icon: Image },
    { id: 'pointcloud', icon: Box },
    { id: 'render', icon: Play },
    { id: 'comparison', icon: GitCompare },
  ]

  return (
    <section id="demo" className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            {t('demo.title')}
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            {t('demo.subtitle')}
          </p>
        </motion.div>

        {/* Tabs */}
        <div className="flex justify-center gap-2 mb-8 flex-wrap">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 sm:px-6 py-3 rounded-xl font-medium transition-all text-sm sm:text-base ${
                activeTab === tab.id
                  ? 'bg-violet-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              <tab.icon size={18} />
              {t(`demo.tabs.${tab.id}`)}
            </button>
          ))}
        </div>

        {/* Demo Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="relative rounded-2xl overflow-hidden bg-slate-900 border border-slate-800"
          >
            {activeTab === 'input' && (
              <div className="p-6">
                <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                  {Array.from({ length: 24 }).map((_, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.02 }}
                      className="aspect-square rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center text-slate-500 text-xs hover:from-violet-900/50 hover:to-slate-800 transition-colors cursor-pointer"
                    >
                      {i + 1}
                    </motion.div>
                  ))}
                </div>
                <div className="mt-6 text-center">
                  <h4 className="font-semibold text-violet-400">{t('demo.input.title')}</h4>
                  <p className="text-sm text-slate-500 mt-1">{t('demo.input.desc')}</p>
                </div>
              </div>
            )}

            {activeTab === 'pointcloud' && (
              <div className="aspect-video flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
                {/* Simulated 3D point cloud visualization */}
                <div className="absolute inset-0 opacity-30">
                  {Array.from({ length: 100 }).map((_, i) => (
                    <div
                      key={i}
                      className="absolute w-1 h-1 bg-violet-400 rounded-full animate-pulse"
                      style={{
                        left: `${Math.random() * 100}%`,
                        top: `${Math.random() * 100}%`,
                        animationDelay: `${Math.random() * 2}s`,
                      }}
                    />
                  ))}
                </div>
                <div className="text-center relative z-10">
                  <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-violet-500/20 flex items-center justify-center">
                    <Box size={48} className="text-violet-400" />
                  </div>
                  <h4 className="font-semibold text-lg">{t('demo.pointcloud.title')}</h4>
                  <p className="text-slate-500 text-sm mt-1">{t('demo.pointcloud.desc')}</p>
                  <p className="text-xs text-slate-600 mt-2">~500K points</p>
                </div>
              </div>
            )}

            {activeTab === 'render' && (
              <div className="aspect-video flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900/20 to-slate-900">
                <div className="text-center">
                  <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-blue-500/20 flex items-center justify-center">
                    <Play size={48} className="text-blue-400" />
                  </div>
                  <h4 className="font-semibold text-lg">{t('demo.render.title')}</h4>
                  <p className="text-slate-500 text-sm mt-1">{t('demo.render.desc')}</p>
                  <div className="flex items-center justify-center gap-4 mt-4 text-xs text-slate-500">
                    <span>60 FPS</span>
                    <span>•</span>
                    <span>1080p</span>
                    <span>•</span>
                    <span>Real-time</span>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'comparison' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-6">
                <div className="aspect-video rounded-xl bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center">
                  <div className="text-center">
                    <Image size={32} className="mx-auto mb-2 text-slate-500" />
                    <p className="text-sm text-slate-400">{t('demo.comparison.input')}</p>
                  </div>
                </div>
                <div className="aspect-video rounded-xl bg-gradient-to-br from-violet-900/30 to-slate-800 flex items-center justify-center">
                  <div className="text-center">
                    <Box size={32} className="mx-auto mb-2 text-violet-400" />
                    <p className="text-sm text-violet-300">{t('demo.comparison.output')}</p>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        <p className="text-center text-slate-500 text-sm mt-6">
          Add your images/videos to <code className="px-2 py-1 bg-slate-800 rounded text-xs">web/src/assets/</code>
        </p>
      </div>
    </section>
  )
}