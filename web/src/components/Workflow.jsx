import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { Image, Eye, GitMerge, FileOutput, Box, Monitor, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'

const stepIcons = [Image, Eye, GitMerge, FileOutput, Box, Monitor]

export default function Workflow() {
  const { t } = useTranslation()
  const [expandedStep, setExpandedStep] = useState(null)

  const steps = [
    'input', 'dust3r', 'align', 'colmap', 'train', 'render'
  ]

  return (
    <section className="py-24 relative bg-slate-900/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            {t('workflow.title')}
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            {t('workflow.subtitle')}
          </p>
        </motion.div>

        {/* Workflow Steps */}
        <div className="relative mb-16">
          <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-violet-500/0 via-violet-500/50 to-violet-500/0 -translate-y-1/2" />

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
            {steps.map((step, index) => {
              const Icon = stepIcons[index]
              return (
                <motion.div
                  key={step}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="relative flex flex-col items-center text-center"
                >
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                    className="relative z-10 w-16 h-16 rounded-2xl bg-slate-800 border-2 border-violet-500/50 flex items-center justify-center mb-4 group hover:border-violet-400 transition-colors cursor-pointer"
                    onClick={() => setExpandedStep(expandedStep === index ? null : index)}
                  >
                    <Icon size={28} className="text-violet-400" />
                  </motion.div>

                  <h4 className="font-semibold text-sm mb-1">{t(`workflow.steps.${step}.label`)}</h4>
                  <p className="text-xs text-slate-500">{t(`workflow.steps.${step}.desc`)}</p>

                  <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-violet-600 text-xs flex items-center justify-center font-bold">
                    {index + 1}
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>

        {/* Detailed Steps */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h3 className="text-xl font-bold mb-6 text-center">{t('workflow.details.title')}</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {steps.map((step, index) => (
              <motion.div
                key={step}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.05 }}
                className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 hover:border-violet-500/50 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-violet-500/20 flex items-center justify-center text-violet-400 font-bold text-sm">
                    {index + 1}
                  </div>
                  <div>
                    <h4 className="font-semibold text-sm mb-1">{t(`workflow.details.step${index + 1}.title`)}</h4>
                    <p className="text-xs text-slate-400 leading-relaxed">{t(`workflow.details.step${index + 1}.desc`)}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}