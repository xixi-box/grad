import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { Menu, X, Github, Globe } from 'lucide-react'

export default function Navbar({ scrolled }) {
  const { t, i18n } = useTranslation()
  const [isOpen, setIsOpen] = useState(false)
  const [langOpen, setLangOpen] = useState(false)

  const navLinks = [
    { name: t('nav.features'), href: '#features' },
    { name: t('nav.demo'), href: '#demo' },
    { name: t('nav.benchmark'), href: '#benchmark' },
    { name: t('nav.faq'), href: '#faq' },
    { name: t('nav.quickstart'), href: '#quickstart' },
  ]

  const toggleLang = (lang) => {
    i18n.changeLanguage(lang)
    setLangOpen(false)
  }

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'glass shadow-lg' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <motion.a
            href="#"
            className="flex items-center gap-2 text-xl font-bold"
            whileHover={{ scale: 1.05 }}
          >
            <span className="gradient-text">DUSt3R</span>
            <span className="text-slate-400">→</span>
            <span className="gradient-text">3DGS</span>
          </motion.a>

          {/* Desktop Nav */}
          <div className="hidden lg:flex items-center gap-6">
            {navLinks.map((link) => (
              <a
                key={link.name}
                href={link.href}
                className="text-slate-400 hover:text-slate-50 transition-colors text-sm"
              >
                {link.name}
              </a>
            ))}

            {/* Language Switcher */}
            <div className="relative">
              <button
                onClick={() => setLangOpen(!langOpen)}
                className="flex items-center gap-1 px-3 py-2 text-slate-400 hover:text-slate-50 transition-colors"
              >
                <Globe size={18} />
                <span className="text-sm">{i18n.language === 'zh' ? '中文' : 'EN'}</span>
              </button>
              <AnimatePresence>
                {langOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="absolute right-0 mt-2 w-24 rounded-lg bg-slate-800 border border-slate-700 overflow-hidden"
                  >
                    <button
                      onClick={() => toggleLang('en')}
                      className={`w-full px-4 py-2 text-sm text-left hover:bg-slate-700 ${i18n.language === 'en' ? 'text-violet-400' : 'text-slate-300'}`}
                    >
                      English
                    </button>
                    <button
                      onClick={() => toggleLang('zh')}
                      className={`w-full px-4 py-2 text-sm text-left hover:bg-slate-700 ${i18n.language === 'zh' ? 'text-violet-400' : 'text-slate-300'}`}
                    >
                      中文
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <motion.a
              href="https://github.com/xixi-box/grad"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 rounded-lg transition-colors text-sm"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Github size={16} />
              <span>{t('nav.github')}</span>
            </motion.a>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="lg:hidden p-2 text-slate-400 hover:text-slate-50"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden glass border-t border-slate-700/50"
          >
            <div className="px-4 py-4 space-y-3">
              {navLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  className="block text-slate-400 hover:text-slate-50 transition-colors"
                  onClick={() => setIsOpen(false)}
                >
                  {link.name}
                </a>
              ))}
              <div className="flex gap-2 pt-2 border-t border-slate-700">
                <button
                  onClick={() => toggleLang('en')}
                  className={`px-3 py-1 rounded text-sm ${i18n.language === 'en' ? 'bg-violet-600 text-white' : 'bg-slate-700 text-slate-300'}`}
                >
                  EN
                </button>
                <button
                  onClick={() => toggleLang('zh')}
                  className={`px-3 py-1 rounded text-sm ${i18n.language === 'zh' ? 'bg-violet-600 text-white' : 'bg-slate-700 text-slate-300'}`}
                >
                  中文
                </button>
              </div>
              <a
                href="https://github.com/xixi-box/grad"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2 bg-violet-600 rounded-lg w-fit"
              >
                <Github size={16} />
                <span>GitHub</span>
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}