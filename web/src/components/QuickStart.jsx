import { useState } from 'react'
import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { Copy, Check, Terminal } from 'lucide-react'

function CodeBlock({ title, code }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="rounded-xl bg-slate-900 border border-slate-800 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 bg-slate-800/50 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <Terminal size={16} className="text-violet-400" />
          <span className="text-sm font-medium text-slate-300">{title}</span>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-3 py-1 text-xs rounded-md bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
        >
          {copied ? (
            <>
              <Check size={14} className="text-green-400" />
              Copied!
            </>
          ) : (
            <>
              <Copy size={14} />
              Copy
            </>
          )}
        </button>
      </div>
      <pre className="p-4 text-sm text-slate-300 overflow-x-auto font-mono leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  )
}

export default function QuickStart() {
  const { t } = useTranslation()

  const steps = ['clone', 'install', 'download', 'run', 'train']

  return (
    <section id="quickstart" className="py-24 relative">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            {t('quickstart.title')}
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            {t('quickstart.subtitle')}
          </p>
        </motion.div>

        <div className="space-y-4">
          {steps.map((step, index) => (
            <motion.div
              key={step}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
            >
              <CodeBlock
                title={t(`quickstart.steps.${step}.title`)}
                code={t(`quickstart.steps.${step}.code`)}
              />
            </motion.div>
          ))}
        </div>

        {/* Help Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-8 p-6 rounded-xl bg-violet-500/10 border border-violet-500/20"
        >
          <h3 className="font-semibold mb-2 text-violet-300">{t('quickstart.help.title')}</h3>
          <p className="text-slate-400 text-sm mb-4">
            {t('quickstart.help.desc')}
          </p>
          <a
            href="https://github.com/xixi-box/grad#readme"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-violet-400 hover:text-violet-300 underline underline-offset-2"
          >
            {t('quickstart.help.link')}
          </a>
        </motion.div>
      </div>
    </section>
  )
}