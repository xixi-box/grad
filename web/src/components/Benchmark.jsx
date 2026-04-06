import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { Timer, Cpu, HardDrive, Gauge, MemoryStick, Award } from 'lucide-react'

export default function Benchmark() {
  const { t } = useTranslation()

  const metrics = [
    { key: 'inference', icon: Timer },
    { key: 'alignment', icon: Cpu },
    { key: 'training', icon: HardDrive },
    { key: 'rendering', icon: Gauge },
    { key: 'memory', icon: MemoryStick },
    { key: 'quality', icon: Award },
  ]

  const comparisons = ['colmap', 'nerf', 'instant', 'ours']

  return (
    <section id="benchmark" className="py-24 relative bg-slate-900/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            {t('benchmark.title')}
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            {t('benchmark.subtitle')}
          </p>
        </motion.div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-16">
          {metrics.map((metric, index) => (
            <motion.div
              key={metric.key}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-center"
            >
              <metric.icon size={24} className="mx-auto mb-2 text-violet-400" />
              <div className="text-xl font-bold gradient-text">{t(`benchmark.metrics.${metric.key}.value`)}</div>
              <div className="text-sm font-medium mt-1">{t(`benchmark.metrics.${metric.key}.label`)}</div>
              <div className="text-xs text-slate-500 mt-1">{t(`benchmark.metrics.${metric.key}.desc`)}</div>
            </motion.div>
          ))}
        </div>

        {/* Comparison Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h3 className="text-xl font-bold mb-6 text-center">{t('benchmark.compare.title')}</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-3 px-4 text-slate-400 font-medium">{t('benchmark.compare.headers.method')}</th>
                  <th className="text-center py-3 px-4 text-slate-400 font-medium">{t('benchmark.compare.headers.time')}</th>
                  <th className="text-center py-3 px-4 text-slate-400 font-medium">{t('benchmark.compare.headers.calibration')}</th>
                  <th className="text-center py-3 px-4 text-slate-400 font-medium">{t('benchmark.compare.headers.quality')}</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((row, index) => (
                  <motion.tr
                    key={row}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    className={`border-b border-slate-800 ${row === 'ours' ? 'bg-violet-500/10' : ''}`}
                  >
                    <td className={`py-4 px-4 font-medium ${row === 'ours' ? 'text-violet-400' : ''}`}>
                      {t(`benchmark.compare.rows.${row}.name`)}
                    </td>
                    <td className="py-4 px-4 text-center text-slate-300">{t(`benchmark.compare.rows.${row}.time`)}</td>
                    <td className="py-4 px-4 text-center">
                      <span className={`px-2 py-1 rounded text-xs ${
                        t(`benchmark.compare.rows.${row}.calibration`) === 'No' || t(`benchmark.compare.rows.${row}.calibration`) === '否'
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {t(`benchmark.compare.rows.${row}.calibration`)}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-center text-slate-300">{t(`benchmark.compare.rows.${row}.quality`)}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </section>
  )
}