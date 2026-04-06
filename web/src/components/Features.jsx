import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { Camera, Target, Box, Zap, BarChart3, Monitor } from 'lucide-react'

export default function Features() {
  const { t } = useTranslation()

  const features = [
    { icon: Camera, key: 'depth', color: 'from-pink-500 to-rose-500' },
    { icon: Target, key: 'pose', color: 'from-violet-500 to-purple-500' },
    { icon: Box, key: 'colmap', color: 'from-blue-500 to-cyan-500' },
    { icon: Zap, key: 'render', color: 'from-amber-500 to-orange-500' },
    { icon: BarChart3, key: 'quality', color: 'from-green-500 to-emerald-500' },
    { icon: Monitor, key: 'cross', color: 'from-indigo-500 to-blue-500' },
  ]

  return (
    <section id="features" className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            {t('features.title')}
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            {t('features.subtitle')}
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.key}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ y: -4 }}
              className="group p-6 rounded-2xl bg-slate-900/50 border border-slate-800 hover:border-violet-500/50 transition-all duration-300"
            >
              <div className={`inline-flex p-3 rounded-xl bg-gradient-to-r ${feature.color} mb-4`}>
                <feature.icon size={24} className="text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2 group-hover:text-violet-400 transition-colors">
                {t(`features.items.${feature.key}.title`)}
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                {t(`features.items.${feature.key}.description`)}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}